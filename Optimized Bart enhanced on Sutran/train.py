# train_bart_sutran_data_v3.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
from tqdm import tqdm
import math
from functools import partial
from rapidfuzz.distance import DamerauLevenshtein
import random
import time
import pickle # For loading scalers if saved separately
from torch.utils.data.dataloader import default_collate # Use default collate for dicts

# --- Configuration ---
PROCESSED_TENSOR_PATH = 'processed_tensors_sutran.pt' # Path to saved tensors
METADATA_PATH = 'metadata_sutran.pt'             # Path to saved metadata
CHECKPOINT_DIR = "bart_sutran_checkpoints_v3"    # Specific dir name
BEST_MODEL_PATH = "bart_sutran_best_model_v3.pth" # Specific model name
LOAD_CHECKPOINT_PATH = None # Example: "bart_sutran_checkpoints_v3/bart_sutran_epoch_10.pth"
SAVE_CHECKPOINT_EVERY = 5
EVALUATE_EVERY = 1
LOG_TRANSFORM_APPLIED = True # IMPORTANT: Set according to processdata_sutran_like.py
USE_AMP = torch.cuda.is_available() # Use Automatic Mixed Precision if CUDA is available

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
EPOCHS = 50
GRADIENT_CLIP_NORM = 1.0

# Model Hyperparameters
D_MODEL = 768
N_HEADS = 12
FFN_DIM = 3072
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.1

# --- Dataset for Pre-Tensorized Data ---
class SutranTensorDictDataset(Dataset):
    """Loads data pre-saved as a list of dictionaries."""
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Ensure all tensors are returned
        return self.data[idx]

# --- Model Definition (Adapted BART with Learned Positional Embeddings) ---

class LearnedPositionalEmbedding(nn.Module):
    """Learned Positional Embeddings."""
    def __init__(self, max_position_embeddings, d_model, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(max_position_embeddings, d_model, padding_idx=pad_idx)
        # Initialize weights consistent with transformers
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if pad_idx is not None:
            nn.init.constant_(self.embed.weight[pad_idx], 0)

    def forward(self, input_ids_shape, past_key_values_length: int = 0):
        batch_size, seq_length = input_ids_shape
        device = self.embed.weight.device
        positions = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        return self.embed(positions.expand(batch_size, -1))

# Using standard PyTorch Transformer components for stability
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

# --- Main Adapted BART Model ---
class AdaptedBartModel(nn.Module):
    """Combines feature processing with BART Encoder/Decoder using Learned Positional Embeddings."""
    def __init__(self, metadata, d_model, n_heads, ffn_dim, num_enc_layers, num_dec_layers, dropout):
        super().__init__()
        self.metadata = metadata
        self.d_model = d_model
        self.pad_id = metadata['pad_id']
        self.activity_col = metadata['activity_col'] # Key for activity IDs in batch dict
        self.all_cat_features = metadata.get('all_cat_features', [])
        self.all_num_features = metadata.get('all_num_features', [])
        self.time_label_features = metadata.get('time_label_features', [])
        self.padding_length = metadata['padding_length'] # Max seq length from preprocessing

        # --- Embedding Layers ---
        self.embeddings = nn.ModuleDict()
        total_feature_dim_before_proj = 0
        num_cat_feats = 1 + len(self.all_cat_features)
        num_num_feats = len(self.all_num_features)

        # Dimension allocation strategy
        num_parts = num_cat_feats + (1 if num_num_feats > 0 else 0)
        base_embed_dim = d_model // max(1, num_parts)
        remainder = d_model % max(1, num_parts)

        # Activity Embedding
        act_vocab_info = metadata['feature_vocabs'][self.activity_col]
        act_embed_dim = base_embed_dim + (1 if remainder > 0 else 0); remainder -= (1 if remainder > 0 else 0)
        self.embeddings[self.activity_col] = nn.Embedding(act_vocab_info['size'], act_embed_dim, padding_idx=self.pad_id)
        total_feature_dim_before_proj += act_embed_dim
        print(f"Embedding '{self.activity_col}': Dim={act_embed_dim}, Vocab={act_vocab_info['size']}")

        # Other Categorical Embeddings
        for i, col in enumerate(self.all_cat_features):
            vocab_info = metadata['feature_vocabs'][col]
            embed_dim = base_embed_dim + (1 if i < remainder else 0)
            pad_idx = metadata['feature_padding_ids'].get(col, 0)
            self.embeddings[f'feat_{col}'] = nn.Embedding(vocab_info['size'], embed_dim, padding_idx=pad_idx)
            total_feature_dim_before_proj += embed_dim
            print(f"Embedding 'feat_{col}': Dim={embed_dim}, Vocab={vocab_info['size']}")

        # Numerical Feature Projection
        if num_num_feats > 0:
             num_proj_dim = d_model - total_feature_dim_before_proj
             if num_proj_dim <= 0: num_proj_dim = d_model // 8; print(f"Warn: Embeddings fill d_model. Using num_proj_dim={num_proj_dim}")
             self.numerical_projector = nn.Linear(num_num_feats, num_proj_dim)
             total_feature_dim_before_proj += num_proj_dim
             print(f"Numerical Projection: Dim={num_proj_dim} (from {num_num_feats} features)")
        else: self.numerical_projector = None; print("No numerical features.")

        # --- Input Projection ---
        self.input_proj = nn.Linear(total_feature_dim_before_proj, d_model) if total_feature_dim_before_proj != d_model else nn.Identity()
        print(f"Final Input Projection: {total_feature_dim_before_proj} -> {d_model}")

        # --- Positional Encoding ---
        self.positional_encoding = LearnedPositionalEmbedding(self.padding_length + 1, d_model, self.pad_id)

        # --- Encoder & Decoder ---
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=False) # norm_first=False matches original Transformer
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_enc_layers, norm=nn.LayerNorm(d_model))

        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=False)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_dec_layers, norm=nn.LayerNorm(d_model))

        # --- Output Heads ---
        self.output_proj = nn.Linear(d_model, act_vocab_info['size'])
        num_time_labels = len(self.time_label_features)
        self.time_head = nn.Linear(d_model, num_time_labels) if num_time_labels > 0 else None
        print(f"Output Heads: Activity (size {act_vocab_info['size']}), Time (size {num_time_labels})")

        self.dropout = nn.Dropout(dropout) # Applied after embeddings + pos encoding

    def _prepare_features(self, batch, sequence_key):
        """Combines features for a given sequence key (e.g., 'activities')."""
        combined = []
        base_ids = batch[sequence_key] # (B, L)
        batch_size, seq_len = base_ids.shape
        device = base_ids.device

        # Activity Embedding
        combined.append(self.embeddings[self.activity_col](base_ids))

        # Other Categorical Embeddings
        for col in self.all_cat_features:
            feat_key = f'feat_{col}'
            if feat_key in batch:
                combined.append(self.embeddings[feat_key](batch[feat_key]))
            else: # Add zero tensor if feature missing
                 emb_layer = self.embeddings[feat_key]
                 combined.append(torch.zeros(batch_size, seq_len, emb_layer.embedding_dim, device=device, dtype=emb_layer.weight.dtype))

        # Numerical Features
        if self.numerical_projector:
            num_key = 'num_features'
            if num_key in batch:
                combined.append(self.numerical_projector(batch[num_key]))
            else: # Add zero tensor if feature missing
                 combined.append(torch.zeros(batch_size, seq_len, self.numerical_projector.out_features, device=device, dtype=torch.float32))

        concatenated = torch.cat(combined, dim=-1)
        projected = self.input_proj(concatenated) # Project to d_model
        return projected, base_ids # Return projected features and base_ids for positional encoding base

    @staticmethod
    def _generate_square_subsequent_mask(sz, device):
        """Generates a boolean causal mask for the decoder where True means masked."""
        # Compatible with nn.MultiheadAttention's attn_mask (True means ignore)
        return torch.triu(torch.full((sz, sz), True, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, batch):
        """ Processes batch dictionary. """
        # 1. Prepare Encoder Input
        # Assumes 'activities' key holds the full sequence including SOS and potentially EOS/Padding
        enc_combined_features, enc_act_ids = self._prepare_features(batch, 'activities')
        # Add positional encoding
        enc_input = self.dropout(enc_combined_features + self.positional_encoding(enc_act_ids.shape))
        # Create encoder padding mask (True where padded)
        enc_padding_mask = (batch['activities'] == self.pad_id) # (B, L_src)

        # 2. Encode
        # nn.TransformerEncoder expects src, mask, src_key_padding_mask
        memory = self.encoder(enc_input, src_key_padding_mask=enc_padding_mask) # (B, L_src, D)

        # 3. Prepare Decoder Input (Shifted Right)
        # Input is SOS + target_sequence[:-1]
        dec_input_act_shifted = batch['activities'][:, :-1] # (B, L_target-1)
        # Create a temporary 'shifted' batch dictionary for _prepare_features
        dec_batch_shifted = {self.activity_col: dec_input_act_shifted}
        for col in self.all_cat_features:
             dec_batch_shifted[f'feat_{col}'] = batch[f'feat_{col}'][:, :-1]
        if self.numerical_projector:
             dec_batch_shifted['num_features'] = batch['num_features'][:, :-1, :]

        dec_combined_features, dec_act_ids_for_pos = self._prepare_features(dec_batch_shifted, self.activity_col)
        # Add positional encoding
        dec_input = self.dropout(dec_combined_features + self.positional_encoding(dec_act_ids_for_pos.shape))
        dec_padding_mask = (dec_input_act_shifted == self.pad_id) # Padding mask for decoder input (B, L_target-1)
        L_tgt = dec_input.size(1)
        device = dec_input.device
        # Causal mask for decoder self-attention (L_target-1, L_target-1)
        dec_attn_mask = self._generate_square_subsequent_mask(L_tgt, device)

        # 4. Decode
        # nn.TransformerDecoder expects tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask
        dec_output = self.decoder(
            tgt=dec_input, memory=memory,
            tgt_mask=dec_attn_mask,           # Causal mask
            memory_mask=None,                 # Usually not needed
            tgt_key_padding_mask=dec_padding_mask, # Padding for decoder input
            memory_key_padding_mask=enc_padding_mask # Padding from encoder memory
        ) # (B, L_target-1, D)

        # 5. Output Projections
        activity_logits = self.output_proj(dec_output) # (B, L_target-1, V_act)
        time_preds = self.time_head(dec_output) if self.time_head else None # (B, L_target-1, N_time)

        return activity_logits, time_preds

# --- Multi-Output Loss Function ---
class SutranLikeLoss(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.pad_id = metadata['pad_id']
        self.activity_col = metadata['activity_col']
        self.time_label_features = metadata.get('time_label_features', [])
        self.act_vocab_size = metadata['feature_vocabs'][self.activity_col]['size']
        self.criterion_act = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.criterion_time = nn.L1Loss(reduction='none') # MAE loss per element

    def forward(self, activity_logits, time_preds, batch):
        device = activity_logits.device
        # Targets are shifted left compared to decoder input
        target_act = batch['activity_labels'].to(device) # (B, L_target) - Includes END token
        target_time = batch['time_labels_target'].to(device) # (B, L_target, N_time) - Aligned with target_act

        # Align lengths: predictions have length L_target - 1
        L = activity_logits.size(1)
        target_act = target_act[:, :L]
        target_time = target_time[:, :L, :]

        # Activity Loss
        act_loss = self.criterion_act(activity_logits.reshape(-1, self.act_vocab_size), target_act.reshape(-1))

        # Time Loss
        if time_preds is not None and len(self.time_label_features) > 0:
            time_loss_mask = (target_act != self.pad_id).unsqueeze(-1).expand_as(time_preds) # (B, L, N_time)
            elementwise_time_loss = self.criterion_time(time_preds, target_time)
            masked_time_loss = elementwise_time_loss * time_loss_mask
            avg_time_loss = masked_time_loss.sum() / time_loss_mask.sum().clamp(min=1e-9) # Avoid division by zero
        else: avg_time_loss = torch.tensor(0.0).to(device)

        combined_loss = act_loss + avg_time_loss # Simple sum, can weight time_loss
        return combined_loss, act_loss.item(), avg_time_loss.item()

# --- Evaluation Metrics and Function ---
# Corrected DLS function
def damerau_levenshtein_similarity(p, t):
    # Ensure inputs are sequences (tuple preferred by rapidfuzz)
    if not isinstance(p, (list, tuple, str)):
        p = tuple(map(str, p))
    elif not isinstance(p, tuple):
        p = tuple(p)
    if not isinstance(t, (list, tuple, str)):
        t = tuple(map(str, t))
    elif not isinstance(t, tuple):
        t = tuple(t)

    d = DamerauLevenshtein.distance(p, t)
    m = max(len(p), len(t))
    return 1.0 - (d / m) if m > 0 else 1.0

def mean_absolute_error(p,t,inv=False): # inv=True applies expm1
    pn,tn=np.array(p,dtype=np.float64),np.array(t,dtype=np.float64)
    # Filter NaNs/Infs
    v=np.isfinite(pn)&np.isfinite(tn)
    pf,tf=pn[v],tn[v]
    if len(pf)==0: return 0.0
    # Apply inverse transform if needed
    if inv: # Apply inverse log1p
        pi,ti = np.expm1(np.maximum(pf,-30)), np.expm1(np.maximum(tf,-30)) # Clamp before expm1
    else:
        pi,ti = pf,tf
    # Calculate MAE
    return np.mean(np.abs(pi-ti))

def evaluate_model(model, dataloader, device, loss_fn, metadata):
    """Evaluates model using teacher forcing, calculates DLS and MAE with inverse transforms."""
    model.eval(); total_loss, total_act_loss, total_time_loss = 0, 0, 0
    all_pred_event_seqs, all_target_event_seqs = [], []
    all_pred_time_lists = {label: [] for label in metadata['time_label_features']}
    all_target_time_lists = {label: [] for label in metadata['time_label_features']}
    num_samples = 0
    scalers = metadata.get('scalers', {})
    try: id_to_event_map = {v: k for k, v in metadata['feature_vocabs'][metadata['activity_col']]['map'].items()}
    except Exception: id_to_event_map = {}; print("Warn: Failed to create id_to_event map")
    pad_id = metadata['pad_id']

    print("Running evaluation...")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            current_batch_size = batch['activities'].size(0)
            num_samples += current_batch_size

            activity_logits, time_preds = model(batch_device)
            loss, act_loss, time_loss = loss_fn(activity_logits, time_preds, batch_device)
            total_loss += loss.item() * current_batch_size; total_act_loss += act_loss * current_batch_size; total_time_loss += time_loss * current_batch_size

            # Prepare for Metrics (CPU)
            target_act = batch['activity_labels'].cpu()
            target_time = batch['time_labels_target'].cpu()
            pred_act_ids = torch.argmax(activity_logits, dim=-1).cpu()
            pred_times = time_preds.cpu() if time_preds is not None else None

            L = min(pred_act_ids.size(1), target_act.size(1))
            pred_act_ids=pred_act_ids[:,:L]; target_act_ids=target_act[:,:L]
            if pred_times is not None: pred_times=pred_times[:,:L,:]; target_times=target_time[:,:L,:]
            else: target_times = None
            target_mask = (target_act_ids != pad_id)

            for i in range(target_act_ids.size(0)):
                actual_len = target_mask[i].sum().item();
                if actual_len == 0: continue
                pi = pred_act_ids[i][:actual_len].tolist(); ti = target_act_ids[i][:actual_len].tolist()
                pe = [id_to_event_map.get(x, f'<UNK_{x}>') for x in pi]
                te = [id_to_event_map.get(x, f'<UNK_{x}>') for x in ti]
                all_pred_event_seqs.append(pe); all_target_event_seqs.append(te)

                if pred_times is not None and target_times is not None:
                    pred_t = pred_times[i][:actual_len, :].numpy(); targ_t = target_times[i][:actual_len, :].numpy()
                    for time_idx, label_name in enumerate(metadata['time_label_features']):
                        pred_vals = pred_t[:, time_idx]; targ_vals = targ_t[:, time_idx]
                        if label_name in scalers:
                             scaler = scalers[label_name]
                             try: pred_vals = scaler.inverse_transform(pred_vals.reshape(-1, 1)).flatten(); targ_vals = scaler.inverse_transform(targ_vals.reshape(-1, 1)).flatten()
                             except Exception as e: print(f"Warn: InvScale {label_name}: {e}")
                        if LOG_TRANSFORM_APPLIED: pred_vals = np.expm1(np.maximum(pred_vals, -30)); targ_vals = np.expm1(np.maximum(targ_vals, -30))
                        all_pred_time_lists[label_name].extend(pred_vals); all_target_time_lists[label_name].extend(targ_vals)

    # Calculate Average Metrics
    avg_loss=total_loss/num_samples if num_samples>0 else 0; avg_act_loss=total_act_loss/num_samples if num_samples>0 else 0; avg_time_loss=total_time_loss/num_samples if num_samples>0 else 0
    mean_dls=np.mean([damerau_levenshtein_similarity(p,t) for p,t in zip(all_pred_event_seqs,all_target_event_seqs)]) if all_pred_event_seqs else 0.0
    mae_results = {}
    for label_name in metadata['time_label_features']:
        mae = mean_absolute_error(all_pred_time_lists[label_name], all_target_time_lists[label_name], inv=False)/60.0 # To min
        mae_results[label_name] = mae

    print(f"\nEvaluation Results:"); print(f"  Avg Loss: {avg_loss:.4f} (Act: {avg_act_loss:.4f}, Time: {avg_time_loss:.4f})")
    print(f"  Activity DLS: {mean_dls:.4f}");
    for label, mae in mae_results.items(): print(f"  {label} MAE (min): {mae:.4f}")
    model.train(); return mean_dls, mae_results

# --- Training Loop Function ---
def run_training(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, metadata,
                 checkpoint_dir, best_model_path, start_epoch=0, initial_best_dls=-1.0, save_every=10, evaluate_every=1):
    print(f"--- Starting Adapted BART Training (Epochs {start_epoch+1}-{epochs}) ---")
    os.makedirs(checkpoint_dir, exist_ok=True); best_dls = initial_best_dls
    # Use updated AMP scaler syntax
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP)

    for epoch in range(start_epoch, epochs):
        model.train(); total_loss=0; total_c_loss=0; total_t_loss=0; num_batches=len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train", leave=False)
        for batch_idx, batch in enumerate(pbar): # Added batch_idx for potential debugging
            try: # Add try-except block for debugging batch issues
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
                    activity_logits, time_preds = model(batch_device)
                    loss, act_loss, time_loss = loss_fn(activity_logits, time_preds, batch_device)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                scaler.step(optimizer); scaler.update()

                total_loss+=loss.item(); total_c_loss+=act_loss; total_t_loss+=time_loss
                pbar.set_postfix({'L':f'{loss.item():.3f}','ActL':f'{act_loss:.3f}','TimeL':f'{time_loss:.3f}'})
            except Exception as e:
                 print(f"\nError during training batch {batch_idx}: {e}")
                 # Optionally print batch details for debugging
                 # for key, value in batch.items():
                 #    if isinstance(value, torch.Tensor): print(f"Batch key '{key}' shape: {value.shape}")
                 #    else: print(f"Batch key '{key}': {value}")
                 # Consider raising the exception again or breaking the loop
                 raise e # Stop training if a batch fails

        avg_loss=total_loss/num_batches if num_batches>0 else 0; avg_c=total_c_loss/num_batches if num_batches>0 else 0; avg_t=total_t_loss/num_batches if num_batches>0 else 0;
        print(f"\nEp {epoch+1} Avg Loss: {avg_loss:.4f} (Act:{avg_c:.4f}, Time:{avg_t:.4f})")

        if (epoch + 1) % evaluate_every == 0:
             current_dls, current_mae_dict = evaluate_model(model, val_loader, device, loss_fn, metadata)
             if current_dls > best_dls:
                 best_dls = current_dls; torch.save(model.state_dict(), best_model_path); print(f"*** New best model saved: DLS {best_dls:.4f} ***")
             else: print(f"DLS {current_dls:.4f} (Best: {best_dls:.4f})")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
             chk_path=os.path.join(checkpoint_dir, f"bart_sutran_epoch_{epoch+1}.pth")
             save_dict = {'epoch':epoch+1, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),
                         'scaler_state_dict':scaler.state_dict(), 'best_dls':best_dls, 'loss':avg_loss}
             torch.save(save_dict, chk_path); print(f"Checkpoint saved: {chk_path}")

    print(f"\n--- Training Complete. Best DLS: {best_dls:.4f} ---"); return best_dls

# --- Main Execution Block ---
if __name__ == "__main__":
    start_run_time = time.time()
    if not os.path.exists(PROCESSED_TENSOR_PATH) or not os.path.exists(METADATA_PATH):
        print(f"Error: Preprocessed data/metadata missing. Run processdata_sutran_like.py."); exit(1)

    print("Loading data and metadata...");
    try: # Load safely
        metadata = torch.load(METADATA_PATH, weights_only=False)
        tensor_data = torch.load(PROCESSED_TENSOR_PATH, weights_only=False)
        if 'scalers' not in metadata and os.path.exists('scalers.pkl'):
            try:
                with open('scalers.pkl', 'rb') as f: metadata['scalers'] = pickle.load(f); print("Loaded scalers from scalers.pkl")
            except Exception as e: print(f"Could not load scalers.pkl: {e}"); metadata['scalers'] = {}
        elif 'scalers' not in metadata: metadata['scalers'] = {}
    except Exception as e: print(f"Error loading data files: {e}"); exit(1)
    print("Data and metadata loaded.")

    # Create Datasets & Split
    train_val_dataset = SutranTensorDictDataset(tensor_data['train'])
    val_frac = 0.2
    if len(train_val_dataset) >= 5:
        val_size = int(val_frac * len(train_val_dataset)); train_size = len(train_val_dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    else: train_ds=train_val_dataset; val_ds=SutranTensorDictDataset(tensor_data['test']); print(f"Warning: Using test set for validation.")
    print(f"Train size: {len(train_ds)}, Validation size: {len(val_ds)}")

    # Create DataLoaders
    # Set num_workers=0 for debugging if training gets stuck
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=default_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=default_collate)
    print("DataLoaders created.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")

    # Instantiate Adapted Model
    model = AdaptedBartModel(metadata, D_MODEL, N_HEADS, FFN_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT).to(device)
    print(f"AdaptedBartModel instantiated ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params).")

    optimizer=optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=0.01)
    loss_fn=SutranLikeLoss(metadata).to(device)
    # Use updated AMP scaler syntax
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP)

    start_epoch=0; best_dls=-1.0
    # Checkpoint loading
    if LOAD_CHECKPOINT_PATH and os.path.isfile(LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint: {LOAD_CHECKPOINT_PATH}");
        try:
            chkpt=torch.load(LOAD_CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(chkpt['model_state_dict']); print("Model state loaded.")
            if 'optimizer_state_dict' in chkpt:
                 try: optimizer.load_state_dict(chkpt['optimizer_state_dict']); print("Optimizer loaded.")
                 except: print("Warn: Optimizer load failed.")
            if 'scaler_state_dict' in chkpt and USE_AMP:
                 scaler.load_state_dict(chkpt['scaler_state_dict']); print("Scaler loaded.")
            start_epoch=chkpt.get('epoch',0); best_dls=chkpt.get('best_dls',-1.0);
            print(f"Resuming ep {start_epoch+1}. Best DLS: {best_dls:.4f}")
        except Exception as e: print(f"Error loading checkpoint: {e}. Starting fresh."); start_epoch=0; best_dls=-1.0
    else: print("Starting training from scratch.")

    # --- Run Training ---
    run_training(model, train_loader, val_loader, optimizer, loss_fn, device, EPOCHS, metadata,
                 CHECKPOINT_DIR, BEST_MODEL_PATH, start_epoch, best_dls, SAVE_CHECKPOINT_EVERY, EVALUATE_EVERY)

    end_run_time = time.time()
    print(f"--- Training Script Finished ({ (end_run_time - start_run_time)/60:.2f} minutes) ---")