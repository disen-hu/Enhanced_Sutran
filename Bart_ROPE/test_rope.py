# test_rope.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import math
from functools import partial
from rapidfuzz.distance import DamerauLevenshtein

# --- Configuration ---
PROCESSED_DATA_PATH = 'processed_data_rope.pt'
VOCAB_INFO_PATH = 'vocab_info_rope.pt'
BEST_MODEL_PATH = "rope_best_model.pth" # Path to the saved RoPE best model
BATCH_SIZE = 64
MAX_SEQ_LENGTH_MODEL = 1024 # Must match training model capacity
MAX_SEQ_LENGTH_LOADER = 128 # Must match training loader max length

# Model Hyperparameters (MUST match the trained RoPE model)
D_MODEL = 512
N_HEADS = 8
FFN_DIM = 2048
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 3
ROPE_BASE = 10000

# --- RoPE Implementation (Copied) ---
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2); x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
    if freqs_cos.dim() == 2:
        if len(x.shape)==4: f_cos,f_sin=freqs_cos.unsqueeze(0).unsqueeze(1),freqs_sin.unsqueeze(0).unsqueeze(1)
        elif len(x.shape)==3: f_cos,f_sin=freqs_cos.unsqueeze(0),freqs_sin.unsqueeze(0)
        else: f_cos,f_sin=freqs_cos,freqs_sin # Assume broadcastable
    else: f_cos,f_sin=freqs_cos,freqs_sin
    r_x1=x1*f_cos-x2*f_sin; r_x2=x1*f_sin+x2*f_cos; r_x=torch.stack((r_x1,r_x2),dim=-1).flatten(-2); return r_x.type_as(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: int = 10000):
        super().__init__(); assert dim % 2 == 0; self.dim=dim; self.max_seq_len=max_seq_len; self.base=base
        inv_freq=1.0/(self.base**(torch.arange(0,self.dim,2).float()/self.dim)); t=torch.arange(self.max_seq_len,dtype=inv_freq.dtype); freqs=torch.outer(t,inv_freq)
        self.register_buffer("freqs_cos",freqs.cos(),persistent=False); self.register_buffer("freqs_sin",freqs.sin(),persistent=False)
    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len: seq_len = self.max_seq_len
        return self.freqs_cos[:seq_len,:].to(device), self.freqs_sin[:seq_len,:].to(device)

# --- Model Definition (RoPE Enhanced - Copied) ---
def gelu(x): return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class BartAttention(nn.Module): # Simplified version from training for clarity
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__(); assert d_model % n_heads == 0; self.d_model=d_model; self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.q_proj=nn.Linear(d_model,d_model); self.k_proj=nn.Linear(d_model,d_model); self.v_proj=nn.Linear(d_model,d_model); self.out_proj=nn.Linear(d_model,d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, query, key, value, freqs_cos=None, freqs_sin=None, attn_mask=None, key_padding_mask=None):
        L_q, B, _ = query.size(); L_k = key.size(0)
        Q = self.q_proj(query).view(L_q, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        K = self.k_proj(key).view(L_k, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        V = self.v_proj(value).view(L_k, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        if freqs_cos is not None: Q=apply_rotary_pos_emb(Q,freqs_cos[:L_q],freqs_sin[:L_q]); K=apply_rotary_pos_emb(K,freqs_cos[:L_k],freqs_sin[:L_k])
        combined_mask=None;
        if key_padding_mask is not None: combined_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        if attn_mask is not None: mask_expanded = attn_mask.bool().unsqueeze(0).unsqueeze(1); combined_mask = combined_mask | mask_expanded if combined_mask is not None else mask_expanded
        attn_output = nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=combined_mask, dropout_p=0.0) # No dropout in eval
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.d_model).transpose(0,1); return self.out_proj(attn_output)
class BartEncoderLayer(nn.Module): # Simplified
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__(); self.self_attn=BartAttention(d_model,n_heads,dropout); self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model)
        self.fc1=nn.Linear(d_model,ffn_dim); self.fc2=nn.Linear(ffn_dim,d_model); self.dropout=nn.Dropout(dropout) # Dropout obj needed but inactive in eval
    def forward(self, x, freqs_cos, freqs_sin, src_key_padding_mask=None):
        a=self.self_attn(x,x,x,freqs_cos,freqs_sin,key_padding_mask=src_key_padding_mask); x=self.norm1(x+a) # No dropout in eval
        f=self.fc2(gelu(self.fc1(x))); x=self.norm2(x+f); return x
class BartDecoderLayer(nn.Module): # Simplified
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__(); self.self_attn=BartAttention(d_model,n_heads,dropout); self.cross_attn=BartAttention(d_model,n_heads,dropout)
        self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model); self.norm3=nn.LayerNorm(d_model)
        self.fc1=nn.Linear(d_model,ffn_dim); self.fc2=nn.Linear(ffn_dim,d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, x, memory, freqs_cos, freqs_sin, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        s=self.self_attn(x,x,x,freqs_cos,freqs_sin,tgt_mask,tgt_key_padding_mask); x=self.norm1(x+s)
        c=self.cross_attn(x,memory,memory,key_padding_mask=memory_key_padding_mask); x=self.norm2(x+c)
        f=self.fc2(gelu(self.fc1(x))); x=self.norm3(x+f); return x
class BartEncoder(nn.Module): # Simplified
    def __init__(self,vocab_size,d_model,n_heads,ffn_dim,num_layers,pad_id,rotary_emb,dropout=0.1):
        super().__init__(); self.embed_tokens=nn.Embedding(vocab_size,d_model,padding_idx=pad_id); self.rotary_emb=rotary_emb
        self.layers=nn.ModuleList([BartEncoderLayer(d_model,n_heads,ffn_dim,dropout) for _ in range(num_layers)]); self.layer_norm=nn.LayerNorm(d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, src_tokens, src_key_padding_mask=None):
        seq_len=src_tokens.size(1); x=self.embed_tokens(src_tokens); x=x.transpose(0,1) # No dropout in eval
        f_cos,f_sin=self.rotary_emb(seq_len,x.device);
        for l in self.layers: x=l(x,f_cos,f_sin,src_key_padding_mask)
        x=self.layer_norm(x); return x
class BartDecoder(nn.Module): # Simplified
    def __init__(self,vocab_size,d_model,n_heads,ffn_dim,num_layers,pad_id,rotary_emb,dropout=0.1):
        super().__init__(); self.embed_tokens=nn.Embedding(vocab_size,d_model,padding_idx=pad_id); self.rotary_emb=rotary_emb
        self.layers=nn.ModuleList([BartDecoderLayer(d_model,n_heads,ffn_dim,dropout) for _ in range(num_layers)]); self.layer_norm=nn.LayerNorm(d_model); self.dropout=nn.Dropout(dropout)
    def _generate_square_subsequent_mask(self,sz,dev): return torch.triu(torch.full((sz,sz),True,dtype=torch.bool,device=dev),diagonal=1)
    def forward(self, tgt_tokens, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        seq_len=tgt_tokens.size(1); x=self.embed_tokens(tgt_tokens); x=x.transpose(0,1) # No dropout in eval
        f_cos,f_sin=self.rotary_emb(seq_len,x.device); tgt_mask=self._generate_square_subsequent_mask(seq_len,x.device)
        for l in self.layers: x=l(x,memory,f_cos,f_sin,tgt_mask,memory_key_padding_mask,tgt_key_padding_mask)
        x=self.layer_norm(x); return x.transpose(0,1)
class BartModel(nn.Module): # Simplified
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim, num_encoder_layers, num_decoder_layers, pad_id, rope_base, dropout=0.1): # Dropout value unused in eval
        super().__init__(); head_dim=d_model//n_heads; assert head_dim%2==0
        self.rotary_emb=RotaryEmbedding(dim=head_dim,max_seq_len=max_positions,base=rope_base)
        self.encoder=BartEncoder(vocab_size,d_model,n_heads,ffn_dim,num_encoder_layers,pad_id,self.rotary_emb,dropout=0.0) # Force dropout 0 in eval
        self.decoder=BartDecoder(vocab_size,d_model,n_heads,ffn_dim,num_decoder_layers,pad_id,self.rotary_emb,dropout=0.0) # Force dropout 0 in eval
        self.output_proj=nn.Linear(d_model,vocab_size); self.time_head=nn.Linear(d_model,1)
    def forward(self, src_tokens, tgt_tokens, src_key_padding_mask=None, tgt_key_padding_mask=None):
        mem=self.encoder(src_tokens,src_key_padding_mask=src_key_padding_mask)
        dec=self.decoder(tgt_tokens,mem,memory_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        logits=self.output_proj(dec); time_pred=self.time_head(dec).squeeze(-1); return logits,time_pred

# --- Dataset, Collator, Metrics (Copied) ---
class ProcessedSeqDataset(Dataset):
    def __init__(self, sequences, deltas): self.sequences = sequences; self.deltas = deltas
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.deltas[idx]
def collate_fn(batch, pad_id, max_len=None):
    sequences, deltas = zip(*batch);
    if max_len is not None: sequences=[s[:max_len] for s in sequences]; deltas=[d[:max_len] for d in deltas]
    max_len_in_batch = max(len(s) for s in sequences); padded_seq=[]; padded_dts=[]
    for seq, dts in zip(sequences, deltas):
        pad_len = max_len_in_batch - len(seq); padded_seq.append(seq + [pad_id]*pad_len); padded_dts.append(list(dts) + [0.0]*pad_len)
    padded_seq=torch.tensor(padded_seq,dtype=torch.long); padded_dts=torch.tensor(padded_dts,dtype=torch.float); key_padding_mask=(padded_seq==pad_id)
    return {'tokens':padded_seq,'deltas':padded_dts,'key_padding_mask':key_padding_mask}
def damerau_levenshtein_similarity(p,t):
    if not isinstance(p,(list,str)): p=tuple(map(str,p)); else: p=tuple(p)
    if not isinstance(t,(list,str)): t=tuple(map(str,t)); else: t=tuple(t)
    d=DamerauLevenshtein.distance(p,t); m=max(len(p),len(t)); return 1.0-(d/m) if m>0 else 1.0
def mean_absolute_error(p,t,inv=False):
    pn,tn=np.array(p,dtype=np.float64),np.array(t,dtype=np.float64); v=np.isfinite(pn)&np.isfinite(tn); pf,tf=pn[v],tn[v];
    if len(pf)==0: return 0.0;
    if inv: pi,ti=np.expm1(np.maximum(pf,-1e6)),np.expm1(np.maximum(tf,-1e6)); else: pi,ti=pf,tf;
    return np.mean(np.abs(pi-ti))

# --- Final Test Evaluation Function ---
def run_test_evaluation(model, dataloader, device, id_to_event, pad_id):
    model.eval(); preds_e=[]; targs_e=[]; preds_d=[]; targs_d=[]
    print("--- Running Final Test Evaluation (RoPE Model) ---")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing", leave=False)
        for batch in pbar:
            tok=batch['tokens'].to(device); dlt=batch['deltas']; mask=batch['key_padding_mask'].to(device); b_size=tok.size(0)
            t_in=tok[:,:-1]; m_in=mask[:,:-1]; t_out=tok[:,1:] # Ground truth for comparison
            # NOTE: Using ground truth t_in for single-step prediction eval, same as during training eval.
            # For true generative eval, would need autoregressive loop here.
            logits, t_pred = model(tok,t_in,mask,m_in)
            p_ids=torch.argmax(logits,dim=-1).cpu(); p_dlt=t_pred.cpu().numpy(); t_ids=t_out.cpu(); t_dlt=dlt[:,1:].numpy(); t_mask=mask[:,1:].cpu()
            for i in range(b_size):
                l=(~t_mask[i]).sum().item();
                if l==0: continue
                pi=p_ids[i][:l].tolist(); ti=t_ids[i][:l].tolist(); pd=p_dlt[i][:l]; td=t_dlt[i][:l]
                pe=[id_to_event.get(x,'<unk>') for x in pi]; te=[id_to_event.get(x,'<unk>') for x in ti]
                preds_e.append(pe); targs_e.append(te); preds_d.extend(pd); targs_d.extend(td)
    dls_s=[damerau_levenshtein_similarity(p,t) for p,t in zip(preds_e,targs_e)]; m_dls=np.mean(dls_s) if dls_s else 0.0
    m_mae=mean_absolute_error(preds_d,targs_d,inv=True)/60.0 if preds_d else 0.0
    print("\n--- Final Test Results (RoPE Model) ---"); print(f"  DLS: {m_dls:.4f}"); print(f"  MAE: {m_mae:.4f} (min)")
    return m_dls, m_mae

# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(VOCAB_INFO_PATH): print(f"Error: Preprocessed data/vocab missing. Run preprocess_data_rope.py."); exit(1)
    if not os.path.exists(BEST_MODEL_PATH): print(f"Error: Best model {BEST_MODEL_PATH} not found. Run train_rope.py."); exit(1)

    print("Loading data/vocab..."); data=torch.load(PROCESSED_DATA_PATH); vocab_info=torch.load(VOCAB_INFO_PATH); print("Loaded.")
    test_seqs=data['test_seqs']; test_deltas=data['test_deltas']; vocab_size=vocab_info['vocab_size']; pad_id=vocab_info['pad_id']; id_to_event=vocab_info['id_to_event']

    print("Creating Test DataLoader..."); test_ds=ProcessedSeqDataset(test_seqs,test_deltas)
    coll_test=partial(collate_fn,pad_id=pad_id,max_len=MAX_SEQ_LENGTH_LOADER)
    test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,collate_fn=coll_test,num_workers=4,pin_memory=True); print("DataLoader created.")

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    model=BartModel(vocab_size, MAX_SEQ_LENGTH_MODEL, D_MODEL, N_HEADS, FFN_DIM, NUM_ENC_LAYERS, NUM_DEC_LAYERS, pad_id, ROPE_BASE, dropout=0.0).to(device) # Dropout=0.0 for eval
    print("Model architecture loaded.")

    try: model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device)); print(f"Loaded best weights: {BEST_MODEL_PATH}")
    except Exception as e: print(f"Error loading weights from {BEST_MODEL_PATH}: {e}"); exit(1)

    run_test_evaluation(model, test_loader, device, id_to_event, pad_id)
    print("--- Testing Script Finished ---")