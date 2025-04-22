# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import math
from functools import partial
from rapidfuzz.distance import DamerauLevenshtein # For DLS calculation

# --- Configuration ---
PROCESSED_DATA_PATH = 'processed_data.pt'
VOCAB_INFO_PATH = 'vocab_info.pt'
CHECKPOINT_DIR = "checkpoints"     # Directory to save training checkpoints
BEST_MODEL_PATH = "best_model.pth" # Path to save the best model weights
LOAD_CHECKPOINT_PATH = None # Example: "checkpoints/model_epoch_10.pth" to resume
SAVE_CHECKPOINT_EVERY = 5   # Save a checkpoint every N epochs
EVALUATE_EVERY = 1          # Evaluate on test set every N epochs

# Training Hyperparameters
BATCH_SIZE = 64       # Adjust based on GPU memory
LEARNING_RATE = 3e-5
EPOCHS = 50           # Total epochs to train for
MAX_SEQ_LENGTH_DataLoader = 102 # Max length for padding/model (should accommodate SOS/EOS)

# Model Hyperparameters (Should match bart_model.py if defined separately)
D_MODEL = 768
N_HEADS = 12
FFN_DIM = 3072
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.1

# --- Helper Functions / Classes ---

# (Copied from previous bart_model.py for simplicity based on user request)
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, d_model, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(max_position_embeddings, d_model, padding_idx=pad_idx if pad_idx is not None else None)
        self.pad_idx = pad_idx
    def forward(self, input_tensor):
        B, L = input_tensor.size()
        positions = torch.arange(L, dtype=torch.long, device=input_tensor.device).unsqueeze(0).expand(B, L)
        position_embeddings = self.embed(positions)
        return position_embeddings

class BartAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0: raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model; self.n_heads = n_heads; self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model); self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model); self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        L_q, B, _ = query.size(); L_k = key.size(0)
        Q = self.q_proj(query); K = self.k_proj(key); V = self.v_proj(value)
        Q = Q.view(L_q, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(L_k, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(L_k, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))
        if attn_mask is not None: attn_weights = attn_weights + attn_mask
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)
        attn_output = attn_output.transpose(0, 1)
        attn_output = self.out_proj(attn_output)
        return attn_output

class BartEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__(); self.self_attn = BartAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim); self.fc2 = nn.Linear(ffn_dim, d_model)
        self.activation = gelu; self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_key_padding_mask=None):
        residual = x; attn_output = self.self_attn(query=x, key=x, value=x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(residual + self.dropout(attn_output)); residual = x
        ffn_output = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = self.norm2(residual + self.dropout(ffn_output)); return x

class BartDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__(); self.self_attn = BartAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = BartAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim); self.fc2 = nn.Linear(ffn_dim, d_model)
        self.activation = gelu; self.dropout = nn.Dropout(dropout)
    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        residual = x
        self_attn_output = self.self_attn(query=x, key=x, value=x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(residual + self.dropout(self_attn_output)); residual = x
        cross_attn_output = self.cross_attn(query=x, key=memory, value=memory, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(residual + self.dropout(cross_attn_output)); residual = x
        ffn_output = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = self.norm3(residual + self.dropout(ffn_output)); return x

class BartEncoder(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim, num_layers, pad_id, dropout=0.1):
        super().__init__(); self.pad_id = pad_id
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, pad_id)
        self.layers = nn.ModuleList([BartEncoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model); self.dropout = nn.Dropout(dropout)
    def forward(self, src_tokens, src_key_padding_mask=None):
        token_embed = self.embed_tokens(src_tokens); pos_embed = self.embed_positions(src_tokens)
        x = self.dropout(token_embed + pos_embed); x = x.transpose(0, 1)
        for layer in self.layers: x = layer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x); return x

class BartDecoder(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim, num_layers, pad_id, dropout=0.1):
        super().__init__(); self.pad_id = pad_id
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, pad_id)
        self.layers = nn.ModuleList([BartDecoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model); self.dropout = nn.Dropout(dropout)
    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, tgt_tokens, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        token_embed = self.embed_tokens(tgt_tokens); pos_embed = self.embed_positions(tgt_tokens)
        x = self.dropout(token_embed + pos_embed); x = x.transpose(0, 1)
        L_tgt = x.size(0); tgt_mask = self._generate_square_subsequent_mask(L_tgt, x.device)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        x = self.layer_norm(x); x = x.transpose(0, 1); return x

class BartModel(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim,
                 num_encoder_layers, num_decoder_layers, pad_id, dropout=0.1):
        super().__init__()
        self.encoder = BartEncoder(vocab_size, max_positions, d_model, n_heads, ffn_dim, num_encoder_layers, pad_id, dropout)
        self.decoder = BartDecoder(vocab_size, max_positions, d_model, n_heads, ffn_dim, num_decoder_layers, pad_id, dropout)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.time_head = nn.Linear(d_model, 1)
    def forward(self, src_tokens, tgt_tokens, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src_tokens, src_key_padding_mask=src_key_padding_mask)
        dec_out = self.decoder(tgt_tokens, memory, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.output_proj(dec_out)
        time_pred = self.time_head(dec_out).squeeze(-1)
        return logits, time_pred

# --- Dataset and DataLoader ---
class ProcessedSeqDataset(Dataset):
    """Dataset for loading preprocessed sequences and deltas."""
    def __init__(self, sequences, deltas):
        # Data is already encoded lists of numbers
        self.sequences = sequences
        self.deltas = deltas
        # Check if lengths match - they should after preprocessing
        assert len(self.sequences) == len(self.deltas), "Mismatch in number of sequences and deltas"
        for i in range(len(self.sequences)):
             assert len(self.sequences[i]) == len(self.deltas[i]), f"Mismatch length in sequence {i}: seq {len(self.sequences[i])} vs delta {len(self.deltas[i])}"


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return the lists directly
        return self.sequences[idx], self.deltas[idx]

def collate_fn(batch, pad_id, max_len=None):
    """
    Collator function to pad sequences and create masks.
    Args:
        batch (list): List of tuples (sequence, delta).
        pad_id (int): ID of the padding token.
        max_len (int, optional): If provided, sequences longer than this (after SOS/EOS)
                                 will be truncated *during collation*. Defaults to None (pad to max in batch).
    """
    sequences, deltas = zip(*batch)

    # Optional Truncation during collation
    if max_len is not None:
        sequences = [s[:max_len] for s in sequences]
        deltas = [d[:max_len] for d in deltas]

    # Pad to the longest sequence *in the current batch* after potential truncation
    max_len_in_batch = max(len(s) for s in sequences)

    padded_seq = []
    padded_dts = []
    for seq, dts in zip(sequences, deltas):
        seq_padding_len = max_len_in_batch - len(seq)
        dts_padding_len = max_len_in_batch - len(dts) # Should be same as seq_padding_len

        seq_pad = seq + [pad_id] * seq_padding_len
        # Pad deltas with 0.0 (log scale)
        dts_pad = list(dts) + [0.0] * dts_padding_len

        padded_seq.append(seq_pad)
        padded_dts.append(dts_pad)

    padded_seq = torch.tensor(padded_seq, dtype=torch.long)
    padded_dts = torch.tensor(padded_dts, dtype=torch.float)
    key_padding_mask = (padded_seq == pad_id) # True where it's padded

    return {
        'tokens': padded_seq,
        'deltas': padded_dts,
        'key_padding_mask': key_padding_mask
    }

# --- Evaluation Metrics and Function ---
def damerau_levenshtein_similarity(pred_seq, target_seq):
    if not isinstance(pred_seq, (list, str)): pred_seq = list(map(str, pred_seq))
    if not isinstance(target_seq, (list, str)): target_seq = list(map(str, target_seq))
    dl_distance = DamerauLevenshtein.distance(pred_seq, target_seq)
    max_len = max(len(pred_seq), len(target_seq))
    return 1.0 - (dl_distance / max_len) if max_len > 0 else 1.0

def mean_absolute_error(pred, target, inverse_transform=False):
    pred_np = np.array(pred, dtype=np.float64); target_np = np.array(target, dtype=np.float64)
    valid_indices = np.isfinite(pred_np) & np.isfinite(target_np)
    pred_filt = pred_np[valid_indices]; target_filt = target_np[valid_indices]
    if len(pred_filt) == 0: return 0.0
    if inverse_transform:
        pred_inv = np.expm1(np.maximum(pred_filt, -1e6))
        target_inv = np.expm1(np.maximum(target_filt, -1e6))
    else: pred_inv, target_inv = pred_filt, target_filt
    mae = np.mean(np.abs(pred_inv - target_inv)); return mae

def evaluate_model(model, dataloader, device, ce_loss_fn, mse_loss_fn, id_to_event, pad_id):
    """Evaluates the model (used during training)."""
    model.eval()
    total_concept_loss = 0; total_time_loss = 0
    all_pred_event_seqs = []; all_target_event_seqs = []
    all_pred_delta_seqs = []; all_target_delta_seqs = []
    num_samples = 0

    print("Running evaluation...")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            tokens = batch['tokens'].to(device); deltas = batch['deltas']
            key_padding_mask = batch['key_padding_mask'].to(device)
            batch_size = tokens.size(0)
            num_samples += batch_size

            eval_tgt_tokens = tokens[:, :-1]; eval_tgt_mask = key_padding_mask[:, :-1]
            logits, time_pred = model(tokens, eval_tgt_tokens, key_padding_mask, eval_tgt_mask)

            target_labels_for_loss = tokens[:, 1:].contiguous().view(-1)
            target_deltas_for_loss = deltas[:, 1:].contiguous()
            pred_logits_for_loss = logits.contiguous().view(-1, logits.size(-1))
            pred_times_for_loss = time_pred.contiguous()

            concept_loss = ce_loss_fn(pred_logits_for_loss, target_labels_for_loss.to(device))
            time_loss = mse_loss_fn(pred_times_for_loss, target_deltas_for_loss.to(device))
            total_concept_loss += concept_loss.item() * batch_size
            total_time_loss += time_loss.item() * batch_size

            pred_token_ids = torch.argmax(logits, dim=-1).cpu()
            pred_deltas_np = time_pred.cpu().numpy()
            target_token_ids = tokens[:, 1:].cpu()
            target_deltas_np = deltas[:, 1:].numpy()
            target_padding_mask = key_padding_mask[:, 1:].cpu()

            for i in range(batch_size):
                actual_len = (~target_padding_mask[i]).sum().item()
                if actual_len == 0: continue
                pred_ids_seq = pred_token_ids[i][:actual_len].tolist()
                target_ids_seq = target_token_ids[i][:actual_len].tolist()
                pred_delta_seq = pred_deltas_np[i][:actual_len]
                target_delta_seq = target_deltas_np[i][:actual_len]
                pred_event_seq = [id_to_event.get(idx, '<unk>') for idx in pred_ids_seq]
                target_event_seq = [id_to_event.get(idx, '<unk>') for idx in target_ids_seq]
                all_pred_event_seqs.append(pred_event_seq)
                all_target_event_seqs.append(target_event_seq)
                all_pred_delta_seqs.extend(pred_delta_seq)
                all_target_delta_seqs.extend(target_delta_seq)

    dls_scores = [damerau_levenshtein_similarity(p, t) for p, t in zip(all_pred_event_seqs, all_target_event_seqs)]
    mean_dls = np.mean(dls_scores) if dls_scores else 0.0
    mean_timestamp_mae = mean_absolute_error(all_pred_delta_seqs, all_target_delta_seqs, inverse_transform=True) / 60.0
    avg_concept_loss = total_concept_loss / num_samples if num_samples > 0 else 0
    avg_time_loss = total_time_loss / num_samples if num_samples > 0 else 0

    print(f"\nEvaluation Results:")
    print(f"  Avg Concept Loss: {avg_concept_loss:.4f} | Avg Time Loss: {avg_time_loss:.4f}")
    print(f"  DLS (Similarity): {mean_dls:.4f} | MAE (Timestamp, min): {mean_timestamp_mae:.4f}")

    model.train()
    return mean_dls, mean_timestamp_mae

# --- Training Loop Function ---
def run_training(model, train_loader, test_loader, optimizer, ce_loss_fn, mse_loss_fn,
                 device, epochs, id_to_event, pad_id,
                 checkpoint_dir, best_model_path,
                 start_epoch=0, initial_best_dls=-1.0, save_every=10, evaluate_every=1):
    """Main training loop."""
    print("--- Starting Model Training ---")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_dls = initial_best_dls
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    print(f"Training from epoch {start_epoch + 1} to {epochs}. Initial best DLS: {best_dls:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0; total_concept_loss = 0; total_time_loss = 0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        for batch in progress_bar:
            src_tokens = batch['tokens'].to(device); tgt_tokens = batch['tokens'].to(device)
            deltas = batch['deltas'].to(device); key_padding_mask = batch['key_padding_mask'].to(device)

            decoder_input_tokens = tgt_tokens[:, :-1]; decoder_target_labels = tgt_tokens[:, 1:]
            decoder_target_deltas = deltas[:, 1:]; decoder_input_mask = key_padding_mask[:, :-1]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, time_pred = model(src_tokens, decoder_input_tokens, key_padding_mask, decoder_input_mask)
                concept_loss = ce_loss_fn(logits.contiguous().view(-1, logits.size(-1)), decoder_target_labels.contiguous().view(-1))
                time_loss = mse_loss_fn(time_pred.contiguous(), decoder_target_deltas.contiguous())
                loss = concept_loss + time_loss # Simple sum, adjust weights if needed

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item(); total_concept_loss += concept_loss.item(); total_time_loss += time_loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'C_L': f'{concept_loss.item():.4f}', 'T_L': f'{time_loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        avg_concept_loss = total_concept_loss / num_batches
        avg_time_loss = total_time_loss / num_batches
        print(f"\nEpoch {epoch+1}/{epochs} finished. Avg Train Loss: {avg_loss:.4f} (C: {avg_concept_loss:.4f}, T: {avg_time_loss:.4f})")

        if (epoch + 1) % evaluate_every == 0:
            current_dls, current_mae = evaluate_model(model, test_loader, device, ce_loss_fn, mse_loss_fn, id_to_event, pad_id)
            if current_dls > best_dls:
                best_dls = current_dls
                torch.save(model.state_dict(), best_model_path)
                print(f"*** New best model saved to {best_model_path} at epoch {epoch+1} with DLS: {best_dls:.4f} ***")
            else:
                print(f"DLS did not improve (Current: {current_dls:.4f}, Best: {best_dls:.4f})")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            checkpoint_save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            save_dict = {
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                'best_dls': best_dls, 'loss': avg_loss
            }
            torch.save(save_dict, checkpoint_save_path)
            print(f"Checkpoint saved to {checkpoint_save_path} at epoch {epoch+1}")

    print(f"\n--- Training complete. Best DLS achieved: {best_dls:.4f} ---")
    print(f"Best model weights saved to: {best_model_path}")
    return best_dls

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Load Preprocessed Data ---
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(VOCAB_INFO_PATH):
        print(f"Error: Preprocessed data ({PROCESSED_DATA_PATH}) or vocab ({VOCAB_INFO_PATH}) not found.")
        print("Please run preprocess_data.py first.")
        exit(1)

    print("Loading preprocessed data...")
    data = torch.load(PROCESSED_DATA_PATH)
    vocab_info = torch.load(VOCAB_INFO_PATH)
    print("Data loaded.")

    train_sequences = data['train_seqs']
    train_deltas = data['train_deltas']
    test_sequences = data['test_seqs']
    test_deltas = data['test_deltas']

    vocab_size = vocab_info['vocab_size']
    pad_id = vocab_info['pad_id']
    id_to_event = vocab_info['id_to_event']

    # --- Create Datasets and DataLoaders ---
    print("Creating Datasets and DataLoaders...")
    train_dataset = ProcessedSeqDataset(train_sequences, train_deltas)
    test_dataset = ProcessedSeqDataset(test_sequences, test_deltas)

    # Use partial to pass pad_id and max_len to collate_fn
    collate_train = partial(collate_fn, pad_id=pad_id, max_len=MAX_SEQ_LENGTH_DataLoader)
    collate_test = partial(collate_fn, pad_id=pad_id, max_len=MAX_SEQ_LENGTH_DataLoader) # Apply same max_len for consistency

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_train, num_workers=4, pin_memory=True)
    # Use test data for evaluation during training
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test, num_workers=4, pin_memory=True)
    print("DataLoaders created.")

    # --- Setup Model, Optimizer, Loss ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BartModel(
        vocab_size=vocab_size,
        max_positions=MAX_SEQ_LENGTH_DataLoader, # Ensure model max_positions matches dataloader max_len
        d_model=D_MODEL, n_heads=N_HEADS, ffn_dim=FFN_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        pad_id=pad_id, dropout=DROPOUT
    ).to(device)
    print(f"Model instantiated with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    mse_loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()) # Scaler needs to be defined here too

    # --- Load Checkpoint if specified ---
    start_epoch = 0
    best_dls_so_far = -1.0
    if LOAD_CHECKPOINT_PATH and os.path.isfile(LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint: {LOAD_CHECKPOINT_PATH}")
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location=device)
        try:
             model.load_state_dict(checkpoint['model_state_dict'])
             print("Model state loaded.")
        except RuntimeError as e:
             print(f"Warning loading model state: {e}. Trying non-strict.")
             model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if 'optimizer_state_dict' in checkpoint:
            try:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 print("Optimizer state loaded.")
            except: print("Warning: Could not load optimizer state.")
        if 'scaler_state_dict' in checkpoint and torch.cuda.is_available():
             scaler.load_state_dict(checkpoint['scaler_state_dict'])
             print("AMP scaler state loaded.")

        start_epoch = checkpoint.get('epoch', 0)
        best_dls_so_far = checkpoint.get('best_dls', -1.0)
        print(f"Resuming from epoch {start_epoch + 1}. Best DLS from checkpoint: {best_dls_so_far:.4f}")
    else:
        print("Starting training from scratch.")

    # --- Run Training ---
    run_training(
        model, train_loader, test_loader, optimizer, ce_loss_fn, mse_loss_fn, device,
        EPOCHS, id_to_event, pad_id, CHECKPOINT_DIR, BEST_MODEL_PATH,
        start_epoch, best_dls_so_far, SAVE_CHECKPOINT_EVERY, EVALUATE_EVERY
    )

    print("--- Training Script Finished ---")