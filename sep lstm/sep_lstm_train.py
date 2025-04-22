# train_sep_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
from tqdm import tqdm
from functools import partial
from rapidfuzz.distance import DamerauLevenshtein
import time

# --- Configuration ---
PROCESSED_DATA_PATH = 'processed_data_rope.pt' # Use the appropriate preprocessed data
VOCAB_INFO_PATH = 'vocab_info_rope.pt'       # Use the appropriate vocab info
CHECKPOINT_DIR = "sep_lstm_checkpoints"
BEST_MODEL_PATH = "sep_lstm_best_model.pth"
LOAD_CHECKPOINT_PATH = None
SAVE_CHECKPOINT_EVERY = 10
EVALUATE_EVERY = 1

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3 # LSTMs often use higher LR than Transformers
EPOCHS = 50
EMBEDDING_DIM = 128  # Dimension for activity embeddings
HIDDEN_DIM = 256     # LSTM hidden state dimension
NUM_LAYERS = 2       # Number of LSTM layers
DROPOUT = 0.2
MAX_SEQ_LENGTH_LOADER = 128 # Max length for padding/truncation in DataLoader

# --- Model Definition: SEP-LSTM ---
class SEP_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.activity_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        # Input to LSTM: activity embedding + time delta (1 dim)
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout_layer = nn.Dropout(dropout)

        # Output heads (predicting the single next step)
        self.fc_activity = nn.Linear(hidden_dim, vocab_size)
        self.fc_time = nn.Linear(hidden_dim, 1) # Predict next time delta

    def forward(self, sequences, deltas, lengths):
        # sequences: (batch, seq_len), includes SOS
        # deltas: (batch, seq_len), includes 0.0 for SOS
        # lengths: (batch,) - actual length of each sequence in the batch (excluding padding)

        # Embed activities
        embedded_activity = self.activity_embedding(sequences) # (batch, seq_len, embed_dim)

        # Combine embeddings and time deltas
        # Unsqueeze time to concatenate: (batch, seq_len, 1)
        time_features = deltas.unsqueeze(-1)
        lstm_input = torch.cat((embedded_activity, time_features), dim=-1) # (batch, seq_len, embed_dim + 1)
        lstm_input = self.dropout_layer(lstm_input)

        # Pack sequence for LSTM efficiency
        packed_input = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack output (optional, we only need the last relevant hidden state)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Get the hidden state of the last *actual* (non-padded) element for each sequence
        # hidden is (num_layers, batch, hidden_dim). We want the output of the last layer.
        last_layer_hidden = hidden[-1] # (batch, hidden_dim)
        last_layer_hidden = self.dropout_layer(last_layer_hidden)

        # Predict next activity and time delta
        activity_logits = self.fc_activity(last_layer_hidden) # (batch, vocab_size)
        time_delta_pred = self.fc_time(last_layer_hidden).squeeze(-1) # (batch,)

        return activity_logits, time_delta_pred

# --- Dataset and DataLoader (Reusable) ---
class ProcessedSeqDataset(Dataset):
    def __init__(self, sequences, deltas): self.sequences = sequences; self.deltas = deltas
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.deltas[idx]

def collate_fn_lstm(batch, pad_id, max_len=None):
    sequences, deltas = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long) # Store original lengths

    if max_len is not None: sequences=[s[:max_len] for s in sequences]; deltas=[d[:max_len] for d in deltas]
    max_len_in_batch = max(len(s) for s in sequences); padded_seq=[]; padded_dts=[]
    for seq, dts in zip(sequences, deltas):
        pad_len = max_len_in_batch - len(seq); padded_seq.append(seq + [pad_id]*pad_len); padded_dts.append(list(dts) + [0.0]*pad_len)
    padded_seq=torch.tensor(padded_seq,dtype=torch.long); padded_dts=torch.tensor(padded_dts,dtype=torch.float)
    key_padding_mask=(padded_seq==pad_id) # Not directly used by LSTM, but useful info

    # Sort by length for potential packing (though enforce_sorted=False handles it)
    lengths, perm_idx = lengths.sort(descending=True)
    padded_seq = padded_seq[perm_idx]
    padded_dts = padded_dts[perm_idx]
    # key_padding_mask = key_padding_mask[perm_idx] # Not strictly needed for SEP-LSTM forward pass

    return {'tokens':padded_seq,'deltas':padded_dts, 'lengths': lengths, 'perm_idx': perm_idx} #'key_padding_mask':key_padding_mask}

# --- Evaluation Metrics and Function (Adapted for SEP-LSTM) ---
def damerau_levenshtein_similarity(p,t): # Compares single elements
    if p == t: return 1.0
    d = DamerauLevenshtein.distance([str(p)], [str(t)]) # Compare as single-item sequences
    return 1.0 - d # Since max_len is 1

def mean_absolute_error(p,t,inv=False):
    pn,tn=np.array(p),np.array(t); v=np.isfinite(pn)&np.isfinite(tn); pf,tf=pn[v],tn[v];
    if len(pf)==0: return 0.0;
    if inv: pi,ti=np.expm1(np.maximum(pf,-1e6)),np.expm1(np.maximum(tf,-1e6)); else: pi,ti=pf,tf;
    return np.mean(np.abs(pi-ti))

def evaluate_model_sep(model, dataloader, device, ce_loss_fn, mse_loss_fn, id_to_event, pad_id):
    model.eval(); total_c_loss=0; total_t_loss=0; preds_e=[]; targs_e=[]; preds_d=[]; targs_d=[]; n_samples=0
    print("Running evaluation (SEP-LSTM)...");
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating SEP", leave=False)
        for batch in pbar:
            tok=batch['tokens'].to(device); dlt=batch['deltas'].to(device); lens=batch['lengths'].to(device); b_size=tok.size(0); n_samples+=b_size
            # Input uses prefix up to second-to-last element
            input_tok = tok[:, :-1]; input_dlt = dlt[:, :-1]; input_lens = lens - 1
            # Target is the single last element (activity and delta)
            target_tok = tok[torch.arange(b_size), lens - 1] # Get last actual token
            target_dlt = dlt[torch.arange(b_size), lens - 1] # Get last actual delta

            # Filter out sequences with length 1 or less (no target)
            valid_indices = input_lens > 0
            if not valid_indices.any(): continue
            input_tok=input_tok[valid_indices]; input_dlt=input_dlt[valid_indices]; input_lens=input_lens[valid_indices]
            target_tok=target_tok[valid_indices]; target_dlt=target_dlt[valid_indices]
            current_batch_size = input_tok.size(0)

            # Forward pass with valid inputs
            logits, t_pred = model(input_tok, input_dlt, input_lens) # Predicts single next step

            # Calculate Loss for the single next step
            c_loss = ce_loss_fn(logits, target_tok)
            t_loss = mse_loss_fn(t_pred, target_dlt)
            total_c_loss+=c_loss.item()*current_batch_size; total_t_loss+=t_loss.item()*current_batch_size

            # Store predictions and targets for metrics
            pred_act_id = torch.argmax(logits, dim=-1).cpu().tolist()
            target_act_id = target_tok.cpu().tolist()
            pred_delta = t_pred.cpu().tolist()
            target_delta = target_dlt.cpu().tolist()

            preds_e.extend([id_to_event.get(x, '<unk>') for x in pred_act_id])
            targs_e.extend([id_to_event.get(x, '<unk>') for x in target_act_id])
            preds_d.extend(pred_delta)
            targs_d.extend(target_delta)

    # Calculate metrics based on single next step predictions
    dls_scores = [damerau_levenshtein_similarity(p, t) for p, t in zip(preds_e, targs_e)]
    mean_dls = np.mean(dls_scores) if dls_scores else 0.0
    mean_mae = mean_absolute_error(preds_d, targs_d, inv=True) / 60.0 if preds_d else 0.0
    avg_c_loss=total_c_loss/n_samples if n_samples>0 else 0; avg_t_loss=total_t_loss/n_samples if n_samples>0 else 0
    print(f"\nSEP Eval Results: C_Loss {avg_c_loss:.4f}, T_Loss {avg_t_loss:.4f} | Next-Step DLS {m_dls:.4f}, Next-Step MAE {m_mae:.4f}")
    model.train(); return mean_dls, mean_mae # Return DLS for best model saving

# --- Training Loop Function ---
def run_training(model, train_loader, test_loader, optimizer, ce_loss_fn, mse_loss_fn, device, epochs, id_to_event, pad_id,
                 checkpoint_dir, best_model_path, start_epoch=0, initial_best_dls=-1.0, save_every=10, evaluate_every=1):
    print(f"--- Starting SEP-LSTM Training (Epochs {start_epoch+1}-{epochs}) ---")
    os.makedirs(checkpoint_dir, exist_ok=True); best_dls = initial_best_dls
    scaler = torch.cuda.amp.GradScaler(enabled=False) # AMP might not be beneficial for LSTMs

    for epoch in range(start_epoch, epochs):
        model.train(); total_loss=0; total_c_loss=0; total_t_loss=0; num_batches=len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train SEP", leave=False)
        for batch in pbar:
            # Prepare inputs and targets for single-step prediction
            tok=batch['tokens'].to(device); dlt=batch['deltas'].to(device); lens=batch['lengths'].to(device); b_size=tok.size(0)
            input_tok=tok[:,:-1]; input_dlt=dlt[:,:-1]; input_lens=lens-1
            target_tok=tok[torch.arange(b_size), lens-1]; target_dlt=dlt[torch.arange(b_size), lens-1]

            # Filter out sequences too short for prediction
            valid_indices = input_lens > 0
            if not valid_indices.any(): continue
            input_tok=input_tok[valid_indices]; input_dlt=input_dlt[valid_indices]; input_lens=input_lens[valid_indices]
            target_tok=target_tok[valid_indices]; target_dlt=target_dlt[valid_indices]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=False): # Disable AMP
                logits, t_pred = model(input_tok, input_dlt, input_lens)
                c_loss = ce_loss_fn(logits, target_tok)
                t_loss = mse_loss_fn(t_pred, target_dlt)
                loss = c_loss + t_loss

            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping might still be useful
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update() # Even if scale is 1.0

            total_loss+=loss.item(); total_c_loss+=c_loss.item(); total_t_loss+=t_loss.item()
            pbar.set_postfix({'L':f'{loss.item():.3f}','CL':f'{c_loss.item():.3f}','TL':f'{t_loss.item():.3f}'})

        avg_loss=total_loss/num_batches if num_batches > 0 else 0; avg_c=total_c_loss/num_batches if num_batches > 0 else 0; avg_t=total_t_loss/num_batches if num_batches > 0 else 0;
        print(f"\nEp {epoch+1} Avg Loss: {avg_loss:.4f} (C:{avg_c:.4f}, T:{avg_t:.4f})")

        if (epoch + 1) % evaluate_every == 0:
            curr_dls, curr_mae = evaluate_model_sep(model, test_loader, device, ce_loss_fn, mse_loss_fn, id_to_event, pad_id)
            if curr_dls > best_dls: best_dls = curr_dls; torch.save(model.state_dict(), best_model_path); print(f"*** Best Next-Step DLS: {best_dls:.4f}. Saved model. ***")
            else: print(f"Next-Step DLS {curr_dls:.4f} (Best: {best_dls:.4f})")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            chk_path=os.path.join(checkpoint_dir, f"sep_lstm_epoch_{epoch+1}.pth"); print(f"Saving chkpt: {chk_path}")
            torch.save({'epoch':epoch+1, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),
                        'scaler_state_dict':scaler.state_dict(), 'best_dls':best_dls, 'loss':avg_loss}, chk_path)

    print(f"\n--- SEP-LSTM Training Complete. Best Next-Step DLS: {best_dls:.4f} ---"); return best_dls

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(VOCAB_INFO_PATH): print(f"Error: Preprocessed data/vocab missing."); exit(1)
    print("Loading preprocessed data..."); data=torch.load(PROCESSED_DATA_PATH); vocab_info=torch.load(VOCAB_INFO_PATH); print("Loaded.")
    train_seqs=data['train_seqs']; train_deltas=data['train_deltas']; test_seqs=data['test_seqs']; test_deltas=data['test_deltas']
    vocab_size=vocab_info['vocab_size']; pad_id=vocab_info['pad_id']; id_to_event=vocab_info['id_to_event']

    print("Creating DataLoaders..."); train_ds=ProcessedSeqDataset(train_seqs,train_deltas); test_ds=ProcessedSeqDataset(test_seqs,test_deltas)
    # Use the LSTM specific collate function
    coll_train=partial(collate_fn_lstm, pad_id=pad_id, max_len=MAX_SEQ_LENGTH_LOADER); coll_test=partial(collate_fn_lstm, pad_id=pad_id, max_len=MAX_SEQ_LENGTH_LOADER)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=coll_train,num_workers=4,pin_memory=True)
    test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,collate_fn=coll_test,num_workers=4,pin_memory=True); print("DataLoaders created.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    model = SEP_LSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id, DROPOUT).to(device)
    print(f"SEP-LSTM Model instantiated ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params).")
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE); ce_loss=nn.CrossEntropyLoss(ignore_index=pad_id); mse_loss=nn.MSELoss()
    scaler=torch.cuda.amp.GradScaler(enabled=False) # Keep scaler object even if disabled

    start_epoch=0; best_dls=-1.0
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
                 CHECKPOINT_DIR,BEST_MODEL_PATH,start_epoch,best_dls,SAVE_CHECKPOINT_EVERY,EVALUATE_EVERY)

    end_time = time.time()
    print(f"--- SEP-LSTM Training Script Finished ---")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")