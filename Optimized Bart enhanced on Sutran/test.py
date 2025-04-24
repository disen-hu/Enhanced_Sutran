# test_sutran_like.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import math
from functools import partial
from rapidfuzz.distance import DamerauLevenshtein
import random

# --- Configuration ---
PROCESSED_TENSOR_PATH = 'processed_tensors_sutran.pt' # Path to saved tensors
METADATA_PATH = 'metadata_sutran.pt'             # Path to saved metadata
BEST_MODEL_PATH = "sutran_like_best_model.pth"   # Path to the best trained model
BATCH_SIZE = 64

# Model Hyperparameters (Must match trained model)
D_MODEL = 768
N_HEADS = 12
FFN_DIM = 3072
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.0 # Set dropout to 0 for evaluation

# --- Dataset Definition (Copied) ---
class SutranTensorDataset(Dataset):
    def __init__(self, data_list): self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- Model Definition (Placeholder - Needs Adaptation, same as in train) ---
# You MUST use the same adapted model definition as used in train_sutran_like.py
# Including BartEncoder, BartDecoder, LearnedPositionalEmbedding/RoPE etc.
# For brevity, it's not repeated here, but assume AdaptedBartModel is defined above.
# --- Placeholder ---
class AdaptedBartModel(nn.Module):
     # PASTE THE FULL AdaptedBartModel DEFINITION FROM train_sutran_like.py HERE
     # Make sure to set dropout to 0.0 in init or during eval
     def __init__(self, metadata, d_model, n_heads, ffn_dim, num_enc_layers, num_dec_layers, dropout):
         super().__init__()
         print("--- WARNING: Using placeholder AdaptedBartModel in test.py ---")
         print("--- Ensure the actual model definition matches train.py ---")
         self.metadata = metadata
         self.d_model = d_model
         self.pad_id = metadata['pad_id']
         self.activity_col = metadata['activity_col']
         act_vocab_size = metadata['feature_vocabs'][self.activity_col]['size']
         num_time_labels = len(metadata['time_label_features'])
         self.dummy_layer = nn.Linear(10, 10) # Minimal layer to allow state_dict loading
         self.output_proj = nn.Linear(d_model, act_vocab_size) # Need this for dummy output shape
         self.time_head = nn.Linear(d_model, num_time_labels) # Need this for dummy output shape


     def forward(self, batch):
         # Dummy forward for placeholder
         prefix_act = batch['activities'][:, :-1]
         dummy_batch_size = prefix_act.size(0)
         dummy_seq_len = prefix_act.size(1)
         dummy_act_logits = torch.randn(dummy_batch_size, dummy_seq_len, self.metadata['feature_vocabs'][self.activity_col]['size']).to(prefix_act.device)
         dummy_time_preds = torch.randn(dummy_batch_size, dummy_seq_len, len(self.metadata['time_label_features'])).to(prefix_act.device)
         return dummy_act_logits, dummy_time_preds

# --- Evaluation Metrics (Copied) ---
def damerau_levenshtein_similarity(p,t):
    if not isinstance(p,(list,str)): p=tuple(map(str,p)); else: p=tuple(p)
    if not isinstance(t,(list,str)): t=tuple(map(str,t)); else: t=tuple(t)
    d=DamerauLevenshtein.distance(p,t); m=max(len(p),len(t)); return 1.0-(d/m) if m>0 else 1.0
def mean_absolute_error(p,t,inv=False):
    pn,tn=np.array(p,dtype=np.float64),np.array(t,dtype=np.float64); v=np.isfinite(pn)&np.isfinite(tn); pf,tf=pn[v],tn[v];
    if len(pf)==0: return 0.0;
    if inv: pi,ti=np.expm1(np.maximum(pf,-1e6)),np.expm1(np.maximum(tf,-1e6)); else: pi,ti=pf,tf;
    return np.mean(np.abs(pi-ti))

# --- Final Test Evaluation Function (Adapt based on new model) ---
def run_test_evaluation(model, dataloader, device, metadata):
    model.eval()
    id_to_event = metadata['feature_vocabs'][metadata['activity_col']]['map'] # Need inverse map or use ids
    id_to_event = {v: k for k, v in id_to_event.items()} # Create inverse map
    pad_id = metadata['pad_id']

    all_pred_event_seqs = []; all_target_event_seqs = []
    all_pred_delta_lists = {label: [] for label in metadata['time_label_features']}
    all_target_delta_lists = {label: [] for label in metadata['time_label_features']}
    scalers = metadata.get('scalers', {}) # Get scalers if saved

    print("--- Running Final Test Evaluation (Sutran-like Data) ---")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing", leave=False)
        for batch in pbar:
            # --- Forward pass using the adapted model ---
            # This assumes model.forward handles the batch dictionary
            activity_logits, time_preds = model(batch)
            # activity_logits: (B, L_pred, V_act)
            # time_preds: (B, L_pred, N_time)

            # --- Get Targets from Batch ---
            # Ensure correct target keys and slicing align with model output length
            target_act = batch['activity_labels'].cpu() # (B, L_target)
            target_time = batch['time_labels_target'].cpu() # (B, L_target, N_time)

            # --- Get Predictions ---
            pred_act_ids = torch.argmax(activity_logits, dim=-1).cpu() # (B, L_pred)
            pred_times = time_preds.cpu()                            # (B, L_pred, N_time)

            # Determine effective length for comparison (min of pred and target length)
            len_pred = pred_act_ids.size(1)
            len_target = target_act.size(1)
            compare_len = min(len_pred, len_target)

            pred_act_ids = pred_act_ids[:, :compare_len]
            target_act_ids = target_act[:, :compare_len]
            pred_times = pred_times[:, :compare_len, :]
            target_times = target_time[:, :compare_len, :]

            # Create mask based on target padding
            target_mask = (target_act_ids != pad_id) # (B, L_compare)


            # --- Collect sequences and times for metrics ---
            for i in range(target_act_ids.size(0)):
                actual_len = target_mask[i].sum().item()
                if actual_len == 0: continue

                # Activity sequences
                pi = pred_act_ids[i][:actual_len].tolist()
                ti = target_act_ids[i][:actual_len].tolist()
                pe = [id_to_event.get(x, '<unk>') for x in pi]
                te = [id_to_event.get(x, '<unk>') for x in ti]
                all_pred_event_seqs.append(pe); all_target_event_seqs.append(te)

                # Time predictions (needs inverse scaling and inverse log transform)
                pred_t = pred_times[i][:actual_len, :].numpy() # (L_actual, N_time)
                targ_t = target_times[i][:actual_len, :].numpy() # (L_actual, N_time)

                for time_idx, label_name in enumerate(metadata['time_label_features']):
                    pred_vals = pred_t[:, time_idx]
                    targ_vals = targ_t[:, time_idx]

                    # Inverse scale if scaler exists
                    if label_name in scalers:
                        scaler = scalers[label_name]
                        pred_vals = scaler.inverse_transform(pred_vals.reshape(-1, 1)).flatten()
                        targ_vals = scaler.inverse_transform(targ_vals.reshape(-1, 1)).flatten()

                    # Inverse log1p if applied during preprocessing (assume all scaled times were log'd)
                    # Need to know if log transform was applied (store in metadata?)
                    # Assuming log transform was applied if scalers exist:
                    if label_name in scalers:
                         pred_vals = np.expm1(np.maximum(pred_vals, -1e6))
                         targ_vals = np.expm1(np.maximum(targ_vals, -1e6))


                    all_pred_delta_lists[label_name].extend(pred_vals)
                    all_target_delta_lists[label_name].extend(targ_vals)


    # --- Calculate Overall Metrics ---
    # DLS for activity sequence
    dls_scores = [damerau_levenshtein_similarity(p, t) for p, t in zip(all_pred_event_seqs, all_target_event_seqs)]
    mean_dls = np.mean(dls_scores) if dls_scores else 0.0

    # MAE for each time prediction (in minutes)
    mae_results = {}
    for label_name in metadata['time_label_features']:
        mae = mean_absolute_error(
            all_pred_delta_lists[label_name],
            all_target_delta_lists[label_name],
            inv=False # Inverse transform already done above
        ) / 60.0 # Convert seconds to minutes
        mae_results[label_name] = mae

    print("\n--- Final Test Results (Sutran-like Data) ---")
    print(f"  Activity DLS: {mean_dls:.4f}")
    for label_name, mae in mae_results.items():
        print(f"  {label_name} MAE (minutes): {mae:.4f}")

    return mean_dls, mae_results

# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(PROCESSED_TENSOR_PATH) or not os.path.exists(METADATA_PATH): print(f"Error: Preprocessed data/metadata missing."); exit(1)
    if not os.path.exists(BEST_MODEL_PATH): print(f"Error: Best model {BEST_MODEL_PATH} not found."); exit(1)

    print("Loading data and metadata..."); tensor_data=torch.load(PROCESSED_TENSOR_PATH); metadata=torch.load(METADATA_PATH); print("Loaded.")

    # Extract test data
    test_data_list = tensor_data['test']

    print("Creating Test DataLoader..."); test_ds=SutranTensorDataset(test_data_list)
    # Use default collate function since dataset returns dicts of tensors
    test_loader=DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True); print("DataLoader created.")

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")

    # --- Instantiate Adapted Model ---
    # !! Replace Placeholder with actual AdaptedBartModel definition from train_sutran_like.py !!
    model = AdaptedBartModel(metadata, D_MODEL, N_HEADS, FFN_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, dropout=0.0).to(device)
    print("Model architecture loaded (ensure definition matches training).")

    # --- Load Best Model Weights ---
    try: model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device)); print(f"Loaded best weights: {BEST_MODEL_PATH}")
    except Exception as e: print(f"Error loading weights from {BEST_MODEL_PATH}: {e}"); exit(1)

    # --- Run Final Evaluation ---
    run_test_evaluation(model, test_loader, device, metadata)

    print("--- Testing Script Finished ---")