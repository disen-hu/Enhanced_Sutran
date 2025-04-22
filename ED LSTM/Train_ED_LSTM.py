# train_ed_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import os
from tqdm import tqdm
import random
from functools import partial
from rapidfuzz.distance import DamerauLevenshtein
import time

# --- Configuration ---
PROCESSED_DATA_PATH = 'processed_data_rope.pt' # Adjust if necessary
VOCAB_INFO_PATH = 'vocab_info_rope.pt'       # Adjust if necessary
CHECKPOINT_DIR = "ed_lstm_checkpoints"
BEST_MODEL_PATH = "ed_lstm_best_model.pth"
LOAD_CHECKPOINT_PATH = None
SAVE_CHECKPOINT_EVERY = 10
EVALUATE_EVERY = 1

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
ENC_LAYERS = 2
DEC_LAYERS = 2 # Can be different from encoder
DROPOUT = 0.2
MAX_SEQ_LENGTH_LOADER = 128 # Max len for dataloader padding/truncation
TEACHER_FORCING_RATIO = 0.5 # Probability of using teacher forcing during training

# --- Model Definition: ED-LSTM ---
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_id, dropout=0.1):
        super().__init__()
        self.activity_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        # Input: activity embedding + time delta
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True) # Use bidirectional encoder
        self.dropout = nn.Dropout(dropout)
        # Project bidirectional hidden/cell states to decoder dimensions if needed
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, sequences, deltas, lengths):
        # Embed activities
        embedded = self.dropout(self.activity_embedding(sequences)) # (batch, seq_len, embed_dim)
        # Combine features
        time_features = deltas.unsqueeze(-1)
        lstm_input = torch.cat((embedded, time_features), dim=-1)

        packed_input = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed_input)
        # hidden: (num_layers*2, batch, hidden_dim), cell: (num_layers*2, batch, hidden_dim)

        # Concatenate forward and backward hidden/cell states from the last layer
        # hidden is L_fwd, L_bwd, L-1_fwd, L-1_bwd ...
        # Get hidden state from last layer (forward and backward)
        last_hidden_fwd = hidden[-2,:,:]
        last_hidden_bwd = hidden[-1,:,:]
        hidden_cat = torch.cat((last_hidden_fwd, last_hidden_bwd), dim=1)

        last_cell_fwd = cell[-2,:,:]
        last_cell_bwd = cell[-1,:,:]
        cell_cat = torch.cat((last_cell_fwd, last_cell_bwd), dim=1)

        # Project to decoder's hidden dimension (assuming decoder is unidirectional)
        hidden_out = torch.tanh(self.fc_hidden(hidden_cat))
        cell_out = torch.tanh(self.fc_cell(cell_cat))

        # Return hidden/cell state repeated for decoder layers if needed (assuming DEC_LAYERS might differ)
        # Here we return the final state of the last bidirectional layer, projected.
        # If DEC_LAYERS > 1, the decoder needs to handle initialization correctly.
        # For simplicity, assume DEC_LAYERS=1 for this projected state or adjust decoder init.
        # Let's return it shaped for a multi-layer decoder (decoder needs to handle it)
        # This assumes decoder hidden_dim matches encoder's projected hidden_dim
        # Shape: (DEC_LAYERS, batch, hidden_dim) - simplistic repetition/zeroing needed if layers differ
        if DEC_LAYERS > 1:
             hidden_out = hidden_out.unsqueeze(0).repeat(DEC_LAYERS, 1, 1)
             cell_out = cell_out.unsqueeze(0).repeat(DEC_LAYERS, 1, 1)
        else:
             hidden_out = hidden_out.unsqueeze(0) # Shape: (1, batch, hidden_dim)
             cell_out = cell_out.unsqueeze(0)   # Shape: (1, batch, hidden_dim)


        return hidden_out, cell_out

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.activity_embedding = nn.Embedding(vocab_size, embedding_dim) # No padding needed for input? Depends.
        # Input: embedded activity + time delta + context? (optional)
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc_activity = nn.Linear(hidden_dim, vocab_size)
        self.fc_time = nn.Linear(hidden_dim, 1)

    def forward(self, input_token, input_delta, hidden, cell):
        # input_token: (batch,) - single token ID for this step
        # input_delta: (batch,) - single delta for this step
        # hidden, cell: (num_layers, batch, hidden_dim) - previous state

        # Embed token, combine features
        embedded = self.dropout(self.activity_embedding(input_token.unsqueeze(1))) # (batch, 1, embed_dim)
        time_features = input_delta.unsqueeze(1).unsqueeze(-1) # (batch, 1, 1)
        lstm_input = torch.cat((embedded, time_features), dim=-1) # (batch, 1, embed_dim + 1)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch, 1, hidden_dim)

        # Predictions
        activity_logits = self.fc_activity(output.squeeze(1)) # (batch, vocab_size)
        time_delta_pred = self.fc_time(output.squeeze(1)).squeeze(-1) # (batch,)

        return activity_logits, time_delta_pred, hidden, cell

class ED_LSTM(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_seq, src_delta, src_lens, trg_seq, trg_delta, teacher_forcing_ratio=0.5):
        # src_seq/delta/lens: Encoder inputs
        # trg_seq/delta: Decoder targets (includes SOS at start)
        batch_size = src_seq.size(0)
        trg_len = trg_seq.size(1) # Max target length (including SOS)
        trg_vocab_size = self.decoder.vocab_size

        # Tensor to store decoder outputs
        outputs_activity = torch.zeros(batch_size, trg_len - 1, trg_vocab_size).to(self.device)
        outputs_time = torch.zeros(batch_size, trg_len - 1).to(self.device)

        # Encode source sequence
        hidden, cell = self.encoder(src_seq, src_delta, src_lens)

        # First input to the decoder is the <sos> token and its 0 delta
        dec_input_token = trg_seq[:, 0]
        dec_input_delta = trg_delta[:, 0]

        # Loop through target sequence (excluding SOS, up to end)
        for t in range(trg_len - 1):
            activity_logits, time_delta_pred, hidden, cell = self.decoder(
                dec_input_token, dec_input_delta, hidden, cell
            )

            # Store predictions
            outputs_activity[:, t, :] = activity_logits
            outputs_time[:, t] = time_delta_pred

            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Get next input token
            top1_activity = activity_logits.argmax(1) # Predicted token
            dec_input_token = trg_seq[:, t+1] if teacher_force else top1_activity

            # Get next input delta (use ground truth if teacher forcing, predicted otherwise)
            # Using ground truth delta even without teacher forcing for simplicity here
            # A better approach might predict delta AND use it, or use a fixed value? Let's use ground truth delta.
            dec_input_delta = trg_delta[:, t+1]
            # Alternative: use predicted time delta if not teacher forcing
            # dec_input_delta = trg_delta[:, t+1] if teacher_force else time_delta_pred

        # outputs_activity: (batch, trg_len-1, vocab_size)
        # outputs_time: (batch, trg_len-1)
        return outputs_activity, outputs_time

# --- Dataset and DataLoader (Reuse ProcessedSeqDataset, use specific collate) ---
# Collator needs to provide source and target sequences separately
def collate_fn_seq2seq(batch, pad_id, max_len=None):
    # Assumes each item in batch is (full_sequence, full_deltas) including SOS/EOS
    # We need to split into src (prefix) and trg (suffix including SOS)
    # Simple split: use full sequence as source, target is also full sequence (decoder ignores last pred)
    sequences, deltas = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)

    if max_len is not None: sequences=[s[:max_len] for s in sequences]; deltas=[d[:max_len] for d in deltas]
    max_len_in_batch = max(len(s) for s in sequences); padded_seq=[]; padded_dts=[]
    for seq, dts in zip(sequences, deltas):
        pad_len = max_len_in_batch - len(seq); padded_seq.append(seq + [pad_id]*pad_len); padded_dts.append(list(dts) + [0.0]*pad_len)

    padded_seq=torch.tensor(padded_seq,dtype=torch.long); padded_dts=torch.tensor(padded_dts,dtype=torch.float)

    # Sort by length for encoder packing
    lengths, perm_idx = lengths.sort(descending=True)
    padded_seq = padded_seq[perm_idx]
    padded_dts = padded_dts[perm_idx]

    # For ED-LSTM: src is the full sequence, trg is also full seq (shifted internally)
    src_seq = padded_seq
    src_delta = padded_dts
    src_lengths = lengths
    trg_seq = padded_seq # Target includes SOS for decoder input shifting
    trg_delta = padded_dts

    return {'src_seq': src_seq, 'src_delta': src_delta, 'src_lengths': src_lengths,
            'trg_seq': trg_seq, 'trg_delta': trg_delta, 'perm_idx': perm_idx}

# --- Evaluation (Teacher Forcing Eval for Simplicity - Needs Autoregressive for Final) ---
def evaluate_model_ed(model, dataloader, device, ce_loss_fn, mse_loss_fn, id_to_event, pad_id):
    model.eval(); total_c_loss=0; total_t_loss=0; preds_e=[]; targs_e=[]; preds_d=[]; targs_d=[]; n_samples=0
    print("Running evaluation (ED-LSTM - Teacher Forcing)...");
    # WARNING: This evaluation uses teacher forcing, not autoregressive generation.
    # Replace with autoregressive decoding for final DLS/MAE suffix evaluation.
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating ED-LSTM (TF)", leave=False)
        for batch in pbar:
            src_seq=batch['src_seq'].to(device); src_delta=batch['src_delta'].to(device); src_lengths=batch['src_lengths'].to(device)
            trg_seq=batch['trg_seq'].to(device); trg_delta=batch['trg_delta'].to(device) # Ground truth targets kept on device for loss
            b_size=src_seq.size(0); n_samples+=b_size

            # Run model with teacher forcing ratio 0 for evaluation (or call a dedicated eval forward)
            # The current forward calculates teacher-forced outputs
            logits, t_pred = model(src_seq, src_delta, src_lengths, trg_seq, trg_delta, teacher_forcing_ratio=0.0) # Use TF=0 to get model's step-by-step prediction based on GT input
            # logits: (batch, trg_len-1, vocab_size), t_pred: (batch, trg_len-1)

            # Targets for loss calculation (exclude SOS)
            target_act = trg_seq[:, 1:].contiguous() # (batch, trg_len-1)
            target_dlt = trg_delta[:, 1:].contiguous() # (batch, trg_len-1)

            # Calculate loss across the sequence, ignoring padding
            c_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), target_act.view(-1))
            loss_mask = (target_act != pad_id).view(-1) # Mask where target is not padding
            t_loss = mse_loss_fn(t_pred.view(-1)[loss_mask], target_dlt.view(-1)[loss_mask]) if loss_mask.any() else torch.tensor(0.0).to(device)
            total_c_loss+=c_loss.item()*b_size; total_t_loss+=t_loss.item()*b_size

            # --- Metrics Calculation (based on teacher-forced step predictions) ---
            # Get predicted IDs and deltas
            pred_act_ids = torch.argmax(logits, dim=-1).cpu() # (batch, trg_len-1)
            pred_deltas = t_pred.cpu().numpy()              # (batch, trg_len-1)
            # Get target IDs and deltas (excluding SOS)
            target_act_ids = target_act.cpu()               # (batch, trg_len-1)
            target_deltas = target_dlt.cpu().numpy()        # (batch, trg_len-1)

            # Iterate through batch to get non-padded sequences for DLS/MAE
            for i in range(b_size):
                # Find actual length of target suffix (excluding SOS, up to EOS or padding)
                actual_len = (target_act_ids[i] != pad_id).sum().item()
                if actual_len == 0: continue

                pi = pred_act_ids[i][:actual_len].tolist()
                ti = target_act_ids[i][:actual_len].tolist()
                pd = pred_deltas[i][:actual_len]
                td = target_deltas[i][:actual_len]

                pe=[id_to_event.get(x,'<unk>') for x in pi]; te=[id_to_event.get(x,'<unk>') for x in ti]
                preds_e.append(pe); targs_e.append(te); preds_d.extend(pd); targs_d.extend(td)

    # Calculate metrics
    # DLS is calculated on the full sequence prediction (using teacher forcing here)
    dls_scores = [damerau_levenshtein_similarity(p, t) for p, t in zip(preds_e, targs_e)]
    mean_dls = np.mean(dls_scores) if dls_scores else 0.0
    # MAE is calculated over all predicted steps (using teacher forcing here)
    mean_mae = mean_absolute_error(preds_d, targs_d, inv=True) / 60.0 if preds_d else 0.0
    avg_c_loss=total_c_loss/n_samples if n_samples>0 else 0; avg_t_loss=total_t_loss/n_samples if n_samples>0 else 0
    print(f"\nED-LSTM Eval (TF): C_Loss {avg_c_loss:.4f}, T_Loss {avg_t_loss:.4f} | Seq DLS {m_dls:.4f}, Step MAE {m_mae:.4f}")
    model.train(); return mean_dls, mean_mae # Return DLS for best model saving

# --- Training Loop Function ---
def run_training(model, train_loader, test_loader, optimizer, ce_loss_fn, mse_loss_fn, device, epochs, id_to_event, pad_id,
                 checkpoint_dir, best_model_path, start_epoch=0, initial_best_dls=-1.0, save_every=10, evaluate_every=1, teacher_forcing_ratio=0.5):
    print(f"--- Starting ED-LSTM Training (Epochs {start_epoch+1}-{epochs}) ---")
    os.makedirs(checkpoint_dir, exist_ok=True); best_dls = initial_best_dls
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for epoch in range(start_epoch, epochs):
        model.train(); total_loss=0; total_c_loss=0; total_t_loss=0; num_batches=len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train ED-LSTM", leave=False)
        for batch in pbar:
            src_seq=batch['src_seq'].to(device); src_delta=batch['src_delta'].to(device); src_lengths=batch['src_lengths'].to(device)
            trg_seq=batch['trg_seq'].to(device); trg_delta=batch['trg_delta'].to(device)
            b_size = src_seq.size(0)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                # Forward pass using teacher forcing according to ratio
                logits, t_pred = model(src_seq, src_delta, src_lengths, trg_seq, trg_delta, teacher_forcing_ratio)
                # logits: (batch, trg_len-1, vocab), t_pred: (batch, trg_len-1)

                # Targets for loss (exclude SOS)
                target_act = trg_seq[:, 1:].contiguous() # (batch, trg_len-1)
                target_dlt = trg_delta[:, 1:].contiguous() # (batch, trg_len-1)

                # Calculate loss across sequence, ignoring padding
                c_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), target_act.view(-1))
                loss_mask = (target_act != pad_id).view(-1)
                t_loss = mse_loss_fn(t_pred.view(-1)[loss_mask], target_dlt.view(-1)[loss_mask]) if loss_mask.any() else torch.tensor(0.0).to(device)
                loss = c_loss + t_loss # Combine loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            total_loss+=loss.item(); total_c_loss+=c_loss.item(); total_t_loss+=t_loss.item()
            pbar.set_postfix({'L':f'{loss.item():.3f}','CL':f'{c_loss.item():.3f}','TL':f'{t_loss.item():.3f}'})

        avg_loss=total_loss/num_batches if num_batches > 0 else 0; avg_c=total_c_loss/num_batches if num_batches > 0 else 0; avg_t=total_t_loss/num_batches if num_batches > 0 else 0;
        print(f"\nEp {epoch+1} Avg Loss: {avg_loss:.4f} (C:{avg_c:.4f}, T:{avg_t:.4f})")

        if (epoch + 1) % evaluate_every == 0:
            # Evaluate using teacher forcing eval function for now
            curr_dls, curr_mae = evaluate_model_ed(model, test_loader, device, ce_loss_fn, mse_loss_fn, id_to_event, pad_id)
            if curr_dls > best_dls: best_dls = curr_dls; torch.save(model.state_dict(), best_model_path); print(f"*** Best Seq DLS (TF Eval): {best_dls:.4f}. Saved model. ***")
            else: print(f"Seq DLS (TF Eval) {curr_dls:.4f} (Best: {best_dls:.4f})")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            chk_path=os.path.join(checkpoint_dir, f"ed_lstm_epoch_{epoch+1}.pth"); print(f"Saving chkpt: {chk_path}")
            torch.save({'epoch':epoch+1, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),
                        'scaler_state_dict':scaler.state_dict(), 'best_dls':best_dls, 'loss':avg_loss}, chk_path)

    print(f"\n--- ED-LSTM Training Complete. Best Seq DLS (TF Eval): {best_dls:.4f} ---"); return best_dls

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(VOCAB_INFO_PATH): print(f"Error: Preprocessed data/vocab missing."); exit(1)
    print("Loading preprocessed data..."); data=torch.load(PROCESSED_DATA_PATH); vocab_info=torch.load(VOCAB_INFO_PATH); print("Loaded.")
    train_seqs=data['train_seqs']; train_deltas=data['train_deltas']; test_seqs=data['test_seqs']; test_deltas=data['test_deltas']
    vocab_size=vocab_info['vocab_size']; pad_id=vocab_info['pad_id']; id_to_event=vocab_info['id_to_event']

    print("Creating DataLoaders..."); train_ds=ProcessedSeqDataset(train_seqs,train_deltas); test_ds=ProcessedSeqDataset(test_seqs,test_deltas)
    coll_train=partial(collate_fn_seq2seq, pad_id=pad_id, max_len=MAX_SEQ_LENGTH_LOADER); coll_test=partial(collate_fn_seq2seq, pad_id=pad_id, max_len=MAX_SEQ_LENGTH_LOADER)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=coll_train,num_workers=4,pin_memory=True)
    test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,collate_fn=coll_test,num_workers=4,pin_memory=True); print("DataLoaders created.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    encoder = EncoderLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, ENC_LAYERS, pad_id, DROPOUT)
    decoder = DecoderLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DEC_LAYERS, DROPOUT) # Assumes DEC_HIDDEN_DIM = ENC_HIDDEN_DIM
    model = ED_LSTM(encoder, decoder, device).to(device)
    print(f"ED-LSTM Model instantiated ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params).")
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE); ce_loss=nn.CrossEntropyLoss(ignore_index=pad_id); mse_loss=nn.MSELoss()
    scaler=torch.cuda.amp.GradScaler(enabled=False)

    start_epoch=0; best_dls=-1.0
    # Checkpoint loading logic (similar to SEP-LSTM)
    if LOAD_CHECKPOINT_PATH and os.path.isfile(LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint: {LOAD_CHECKPOINT_PATH}"); chkpt=torch.load(LOAD_CHECKPOINT_PATH,map_location=device)
        try: model.load_state_dict(chkpt['model_state_dict']); print("Model state loaded.")
        except Exception as e: print(f"Warn model load: {e}.")
        if 'optimizer_state_dict' in chkpt:
            try: optimizer.load_state_dict(chkpt['optimizer_state_dict']); print("Optimizer loaded.")
            except: print("Warn: Optimizer load failed.")
        start_epoch=chkpt.get('epoch',0); best_dls=chkpt.get('best_dls',-1.0); print(f"Resuming ep {start_epoch+1}. Best DLS: {best_dls:.4f}")
    else: print("Starting from scratch.")


    run_training(model,train_loader,test_loader,optimizer,ce_loss,mse_loss,device,EPOCHS,id_to_event,pad_id,
                 CHECKPOINT_DIR,BEST_MODEL_PATH,start_epoch,best_dls,SAVE_CHECKPOINT_EVERY,EVALUATE_EVERY, TEACHER_FORCING_RATIO)

    end_time = time.time()
    print(f"--- ED-LSTM Training Script Finished ---")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")