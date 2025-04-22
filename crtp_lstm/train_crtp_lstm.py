# train_crtp_lstm.py
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
CHECKPOINT_DIR = "crtp_lstm_checkpoints"
BEST_MODEL_PATH = "crtp_lstm_best_model.pth"
LOAD_CHECKPOINT_PATH = None
SAVE_CHECKPOINT_EVERY = 10
EVALUATE_EVERY = 1

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
# CRTP might implicitly use a single LSTM block or separate encoder/decoder
ENC_LAYERS = 1 # Example: Unidirectional encoder
DEC_LAYERS = 2
DROPOUT = 0.2
MAX_SEQ_LENGTH_LOADER = 128
TEACHER_FORCING_RATIO = 0.5

# --- Model Definition: CRTP-LSTM (Structured similar to ED-LSTM for suffix generation) ---
# Using Encoder/Decoder structure allows clear suffix generation
# Differences might be in unidirectionality or state passing, TBD by original paper details
class EncoderLSTM_CRTP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_id, dropout=0.1):
        super().__init__()
        self.activity_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False) # Unidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences, deltas, lengths):
        embedded = self.dropout(self.activity_embedding(sequences))
        time_features = deltas.unsqueeze(-1)
        lstm_input = torch.cat((embedded, time_features), dim=-1)
        packed_input = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed_input)
        # hidden/cell: (num_layers, batch, hidden_dim)
        return hidden, cell

class DecoderLSTM_CRTP(nn.Module):
    # Identical to ED-LSTM decoder for this implementation
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.activity_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc_activity = nn.Linear(hidden_dim, vocab_size)
        self.fc_time = nn.Linear(hidden_dim, 1)

    def forward(self, input_token, input_delta, hidden, cell):
        embedded = self.dropout(self.activity_embedding(input_token.unsqueeze(1)))
        time_features = input_delta.unsqueeze(1).unsqueeze(-1)
        lstm_input = torch.cat((embedded, time_features), dim=-1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        activity_logits = self.fc_activity(output.squeeze(1))
        time_delta_pred = self.fc_time(output.squeeze(1)).squeeze(-1)
        return activity_logits, time_delta_pred, hidden, cell

class CRTP_LSTM(nn.Module):
    # Wrapper similar to ED-LSTM
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # Ensure encoder and decoder hidden sizes match
        assert encoder.lstm.hidden_size == decoder.lstm.hidden_size, "Encoder and Decoder hidden dims must match"
        # Handle layer mismatch if necessary (e.g., repeat/slice encoder state)
        assert encoder.lstm.num_layers <= decoder.lstm.num_layers, "Decoder cannot have fewer layers than encoder for state transfer"


    def forward(self, src_seq, src_delta, src_lens, trg_seq, trg_delta, teacher_forcing_ratio=0.5):
        batch_size = src_seq.size(0); trg_len = trg_seq.size(1); trg_vocab_size = self.decoder.vocab_size
        outputs_activity = torch.zeros(batch_size, trg_len - 1, trg_vocab_size).to(self.device)
        outputs_time = torch.zeros(batch_size, trg_len - 1).to(self.device)

        hidden, cell = self.encoder(src_seq, src_delta, src_lens)

        # Adjust encoder state if layers differ (simple approach: take top layers)
        if self.encoder.lstm.num_layers < self.decoder.lstm.num_layers:
             # Pad with zeros or repeat top layer state - repeating is simpler:
             diff = self.decoder.lstm.num_layers - self.encoder.lstm.num_layers
             hidden = torch.cat([hidden] + [hidden[-1:,:,:]] * diff, dim=0)
             cell = torch.cat([cell] + [cell[-1:,:,:]] * diff, dim=0)
        elif self.encoder.lstm.num_layers > self.decoder.lstm.num_layers:
             # This case is asserted against in __init__
             pass


        dec_input_token = trg_seq[:, 0]; dec_input_delta = trg_delta[:, 0]
        for t in range(trg_len - 1):
            activity_logits, time_delta_pred, hidden, cell = self.decoder(
                dec_input_token, dec_input_delta, hidden, cell)
            outputs_activity[:, t, :] = activity_logits; outputs_time[:, t] = time_delta_pred
            teacher_force = random.random() < teacher_forcing_ratio
            top1_activity = activity_logits.argmax(1)
            dec_input_token = trg_seq[:, t+1] if teacher_force else top1_activity
            dec_input_delta = trg_delta[:, t+1] # Use ground truth delta for next input

        return outputs_activity, outputs_time

# --- Dataset, Collator, Metrics, Eval (Reuse ED-LSTM versions) ---
ProcessedSeqDataset = ProcessedSeqDataset # Already defined in ED-LSTM context
collate_fn_seq2seq = collate_fn_seq2seq # Already defined
damerau_levenshtein_similarity = damerau_levenshtein_similarity # Defined
mean_absolute_error = mean_absolute_error # Defined
evaluate_model_crtp = evaluate_model_ed # Reuse ED evaluation logic (Teacher Forcing)
                                        # Rename for clarity in output needed.

# --- Training Loop (Reuse ED-LSTM version) ---
run_training = run_training # Reuse the training loop function

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
    encoder = EncoderLSTM_CRTP(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, ENC_LAYERS, pad_id, DROPOUT)
    decoder = DecoderLSTM_CRTP(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DEC_LAYERS, DROPOUT)
    model = CRTP_LSTM(encoder, decoder, device).to(device)
    print(f"CRTP-LSTM Model instantiated ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params).")
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE); ce_loss=nn.CrossEntropyLoss(ignore_index=pad_id); mse_loss=nn.MSELoss()
    scaler=torch.cuda.amp.GradScaler(enabled=False)

    start_epoch=0; best_dls=-1.0
    # Checkpoint loading logic
    if LOAD_CHECKPOINT_PATH and os.path.isfile(LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint: {LOAD_CHECKPOINT_PATH}"); chkpt=torch.load(LOAD_CHECKPOINT_PATH,map_location=device)
        try: model.load_state_dict(chkpt['model_state_dict']); print("Model state loaded.")
        except Exception as e: print(f"Warn model load: {e}.")
        if 'optimizer_state_dict' in chkpt:
            try: optimizer.load_state_dict(chkpt['optimizer_state_dict']); print("Optimizer loaded.")
            except: print("Warn: Optimizer load failed.")
        start_epoch=chkpt.get('epoch',0); best_dls=chkpt.get('best_dls',-1.0); print(f"Resuming ep {start_epoch+1}. Best DLS: {best_dls:.4f}")
    else: print("Starting from scratch.")


    # Use the renamed evaluation function if needed, or modify run_training to accept it
    # For now, reusing the ED version implicitly
    run_training(model,train_loader,test_loader,optimizer,ce_loss,mse_loss,device,EPOCHS,id_to_event,pad_id,
                 CHECKPOINT_DIR,BEST_MODEL_PATH,start_epoch,best_dls,SAVE_CHECKPOINT_EVERY,EVALUATE_EVERY, TEACHER_FORCING_RATIO)

    end_time = time.time()
    print(f"--- CRTP-LSTM Training Script Finished ---")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")