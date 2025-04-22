# test.py
import torch
import torch.nn as nn
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
BEST_MODEL_PATH = "best_model.pth" # Path to the saved best model weights
BATCH_SIZE = 64 # Can be larger for testing if memory allows
MAX_SEQ_LENGTH_DataLoader = 102 # Should match the value used during training collation

# Model Hyperparameters (MUST match the trained model)
D_MODEL = 768
N_HEADS = 12
FFN_DIM = 3072
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROPOUT = 0.1 # Dropout is typically inactive in eval mode, but keep for consistency

# --- Helper Functions / Classes (Copied for simplicity) ---
# (Model definition, Dataset, Collator, Metrics - same as in train.py)

def gelu(x): return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, d_model, pad_idx):
        super().__init__(); self.embed = nn.Embedding(max_position_embeddings, d_model, padding_idx=pad_idx if pad_idx is not None else None)
    def forward(self, input_tensor):
        B, L = input_tensor.size(); positions = torch.arange(L, dtype=torch.long, device=input_tensor.device).unsqueeze(0).expand(B, L)
        return self.embed(positions)
class BartAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__(); assert d_model % n_heads == 0; self.d_model=d_model; self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.q_proj=nn.Linear(d_model,d_model); self.k_proj=nn.Linear(d_model,d_model); self.v_proj=nn.Linear(d_model,d_model); self.out_proj=nn.Linear(d_model,d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        L_q,B,_ = query.size(); L_k=key.size(0); Q=self.q_proj(query); K=self.k_proj(key); V=self.v_proj(value)
        Q=Q.view(L_q,B,self.n_heads,self.head_dim).permute(1,2,0,3); K=K.view(L_k,B,self.n_heads,self.head_dim).permute(1,2,0,3); V=V.view(L_k,B,self.n_heads,self.head_dim).permute(1,2,0,3)
        attn_weights = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.head_dim)
        if key_padding_mask is not None: expanded_mask=key_padding_mask.unsqueeze(1).unsqueeze(2); attn_weights=attn_weights.masked_fill(expanded_mask,float('-inf'))
        if attn_mask is not None: attn_weights=attn_weights+attn_mask
        attn_probs=nn.functional.softmax(attn_weights,dim=-1); attn_probs=self.dropout(attn_probs); attn_output=torch.matmul(attn_probs,V)
        attn_output=attn_output.permute(0,2,1,3).contiguous().view(B,L_q,self.d_model); attn_output=attn_output.transpose(0,1); attn_output=self.out_proj(attn_output); return attn_output
class BartEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__(); self.self_attn=BartAttention(d_model,n_heads,dropout=dropout); self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model)
        self.fc1=nn.Linear(d_model,ffn_dim); self.fc2=nn.Linear(ffn_dim,d_model); self.activation=gelu; self.dropout=nn.Dropout(dropout)
    def forward(self, x, src_key_padding_mask=None):
        r=x; a=self.self_attn(x,x,x,key_padding_mask=src_key_padding_mask); x=self.norm1(r+self.dropout(a)); r=x; f=self.fc2(self.dropout(self.activation(self.fc1(x)))); x=self.norm2(r+self.dropout(f)); return x
class BartDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__(); self.self_attn=BartAttention(d_model,n_heads,dropout=dropout); self.cross_attn=BartAttention(d_model,n_heads,dropout=dropout)
        self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model); self.norm3=nn.LayerNorm(d_model); self.fc1=nn.Linear(d_model,ffn_dim); self.fc2=nn.Linear(ffn_dim,d_model); self.activation=gelu; self.dropout=nn.Dropout(dropout)
    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        r=x; s=self.self_attn(x,x,x,attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask); x=self.norm1(r+self.dropout(s)); r=x
        c=self.cross_attn(x,memory,memory,key_padding_mask=memory_key_padding_mask); x=self.norm2(r+self.dropout(c)); r=x
        f=self.fc2(self.dropout(self.activation(self.fc1(x)))); x=self.norm3(r+self.dropout(f)); return x
class BartEncoder(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim, num_layers, pad_id, dropout=0.1):
        super().__init__(); self.embed_tokens=nn.Embedding(vocab_size,d_model,padding_idx=pad_id); self.embed_positions=LearnedPositionalEmbedding(max_positions,d_model,pad_id)
        self.layers=nn.ModuleList([BartEncoderLayer(d_model,n_heads,ffn_dim,dropout) for _ in range(num_layers)]); self.layer_norm=nn.LayerNorm(d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, src_tokens, src_key_padding_mask=None):
        x=self.dropout(self.embed_tokens(src_tokens)+self.embed_positions(src_tokens)); x=x.transpose(0,1)
        for l in self.layers: x=l(x,src_key_padding_mask=src_key_padding_mask)
        x=self.layer_norm(x); return x
class BartDecoder(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim, num_layers, pad_id, dropout=0.1):
        super().__init__(); self.embed_tokens=nn.Embedding(vocab_size,d_model,padding_idx=pad_id); self.embed_positions=LearnedPositionalEmbedding(max_positions,d_model,pad_id)
        self.layers=nn.ModuleList([BartDecoderLayer(d_model,n_heads,ffn_dim,dropout) for _ in range(num_layers)]); self.layer_norm=nn.LayerNorm(d_model); self.dropout=nn.Dropout(dropout)
    def _generate_square_subsequent_mask(self,sz,dev): mask=(torch.triu(torch.ones(sz,sz,device=dev))==1).transpose(0,1); return mask.float().masked_fill(mask==0,float('-inf')).masked_fill(mask==1,float(0.0))
    def forward(self, tgt_tokens, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        x=self.dropout(self.embed_tokens(tgt_tokens)+self.embed_positions(tgt_tokens)); x=x.transpose(0,1)
        L_tgt=x.size(0); tgt_mask=self._generate_square_subsequent_mask(L_tgt,x.device)
        for l in self.layers: x=l(x,memory,tgt_mask=tgt_mask,memory_key_padding_mask=memory_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        x=self.layer_norm(x); x=x.transpose(0,1); return x
class BartModel(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim, num_encoder_layers, num_decoder_layers, pad_id, dropout=0.1):
        super().__init__(); self.encoder=BartEncoder(vocab_size,max_positions,d_model,n_heads,ffn_dim,num_encoder_layers,pad_id,dropout)
        self.decoder=BartDecoder(vocab_size,max_positions,d_model,n_heads,ffn_dim,num_decoder_layers,pad_id,dropout)
        self.output_proj=nn.Linear(d_model,vocab_size); self.time_head=nn.Linear(d_model,1)
    def forward(self, src_tokens, tgt_tokens, src_key_padding_mask=None, tgt_key_padding_mask=None):
        mem=self.encoder(src_tokens,src_key_padding_mask=src_key_padding_mask); dec=self.decoder(tgt_tokens,mem,memory_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        logits=self.output_proj(dec); time_pred=self.time_head(dec).squeeze(-1); return logits,time_pred

class ProcessedSeqDataset(Dataset):
    def __init__(self, sequences, deltas): self.sequences = sequences; self.deltas = deltas
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.deltas[idx]

def collate_fn(batch, pad_id, max_len=None):
    sequences, deltas = zip(*batch)
    if max_len is not None: sequences=[s[:max_len] for s in sequences]; deltas=[d[:max_len] for d in deltas]
    max_len_in_batch = max(len(s) for s in sequences); padded_seq=[]; padded_dts=[]
    for seq, dts in zip(sequences, deltas):
        pad_len = max_len_in_batch - len(seq)
        padded_seq.append(seq + [pad_id]*pad_len); padded_dts.append(list(dts) + [0.0]*pad_len)
    padded_seq=torch.tensor(padded_seq,dtype=torch.long); padded_dts=torch.tensor(padded_dts,dtype=torch.float)
    key_padding_mask=(padded_seq==pad_id); return {'tokens':padded_seq,'deltas':padded_dts,'key_padding_mask':key_padding_mask}

def damerau_levenshtein_similarity(p,t):
    if not isinstance(p,(list,str)): p=list(map(str,p))
    if not isinstance(t,(list,str)): t=list(map(str,t))
    d=DamerauLevenshtein.distance(p,t); m=max(len(p),len(t)); return 1.0-(d/m) if m>0 else 1.0
def mean_absolute_error(p,t,inv=False):
    pn,tn=np.array(p,dtype=np.float64),np.array(t,dtype=np.float64); v=np.isfinite(pn)&np.isfinite(tn); pf,tf=pn[v],tn[v]
    if len(pf)==0: return 0.0
    if inv: pi,ti=np.expm1(np.maximum(pf,-1e6)),np.expm1(np.maximum(tf,-1e6))
    else: pi,ti=pf,tf
    return np.mean(np.abs(pi-ti))

# --- Test Evaluation Function ---
def run_test_evaluation(model, dataloader, device, id_to_event):
    """Evaluates the final model on the test set."""
    model.eval() # Set model to evaluation mode
    all_pred_event_seqs = []; all_target_event_seqs = []
    all_pred_delta_seqs = []; all_target_delta_seqs = []

    print("--- Running Final Test Evaluation ---")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing", leave=False)
        for batch in progress_bar:
            tokens = batch['tokens'].to(device)
            deltas = batch['deltas'] # Keep on CPU
            key_padding_mask = batch['key_padding_mask'].to(device)
            batch_size = tokens.size(0)

            eval_tgt_tokens = tokens[:, :-1]
            eval_tgt_mask = key_padding_mask[:, :-1]

            # Forward pass
            logits, time_pred = model(tokens, eval_tgt_tokens, key_padding_mask, eval_tgt_mask)

            # --- Prepare for Metrics ---
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

    # --- Calculate Overall Metrics ---
    dls_scores = [damerau_levenshtein_similarity(p, t) for p, t in zip(all_pred_event_seqs, all_target_event_seqs)]
    mean_dls = np.mean(dls_scores) if dls_scores else 0.0
    mean_timestamp_mae = mean_absolute_error(all_pred_delta_seqs, all_target_delta_seqs, inv=True) / 60.0 # Inverse transform and convert to minutes

    print("\n--- Final Test Results ---")
    print(f"  DLS (Similarity): {mean_dls:.4f}")
    print(f"  MAE (Next Timestamp, minutes): {mean_timestamp_mae:.4f}")

    return mean_dls, mean_timestamp_mae

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Load Preprocessed Data and Vocab ---
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(VOCAB_INFO_PATH):
        print(f"Error: Preprocessed data ({PROCESSED_DATA_PATH}) or vocab ({VOCAB_INFO_PATH}) not found.")
        print("Please run preprocess_data.py first.")
        exit(1)
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Best model weights ({BEST_MODEL_PATH}) not found.")
        print("Please run train.py first to generate the best model.")
        exit(1)

    print("Loading preprocessed data and vocabulary...")
    data = torch.load(PROCESSED_DATA_PATH)
    vocab_info = torch.load(VOCAB_INFO_PATH)
    print("Data and vocab loaded.")

    # Extract test data and vocab info needed
    test_sequences = data['test_seqs']
    test_deltas = data['test_deltas']
    vocab_size = vocab_info['vocab_size']
    pad_id = vocab_info['pad_id']
    id_to_event = vocab_info['id_to_event']

    # --- Create Test DataLoader ---
    print("Creating Test DataLoader...")
    test_dataset = ProcessedSeqDataset(test_sequences, test_deltas)
    collate_test = partial(collate_fn, pad_id=pad_id, max_len=MAX_SEQ_LENGTH_DataLoader)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test, num_workers=4, pin_memory=True)
    print("Test DataLoader created.")

    # --- Load Model Architecture ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BartModel(
        vocab_size=vocab_size,
        max_positions=MAX_SEQ_LENGTH_DataLoader, # Must match training
        d_model=D_MODEL, n_heads=N_HEADS, ffn_dim=FFN_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        pad_id=pad_id, dropout=DROPOUT # Dropout doesn't matter in eval mode
    ).to(device)
    print("Model architecture loaded.")

    # --- Load Best Model Weights ---
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Successfully loaded best model weights from {BEST_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model weights from {BEST_MODEL_PATH}: {e}")
        exit(1)

    # --- Run Final Evaluation ---
    run_test_evaluation(model, test_loader, device, id_to_event)

    print("--- Testing Script Finished ---")