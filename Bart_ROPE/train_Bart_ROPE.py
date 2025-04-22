# train_rope.py
import torch
import torch.nn as nn
import torch.optim as optim
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
CHECKPOINT_DIR = "rope_checkpoints"       # Directory for RoPE checkpoints
BEST_MODEL_PATH = "rope_best_model.pth"   # Path for RoPE best model
LOAD_CHECKPOINT_PATH = None # Example: "rope_checkpoints/rope_bart_epoch_10.pth"
SAVE_CHECKPOINT_EVERY = 10  # Save frequency
EVALUATE_EVERY = 1         # Evaluate frequency

# Training Hyperparameters
BATCH_SIZE = 64            # Adjust based on GPU memory
LEARNING_RATE = 1e-4       # Example learning rate
EPOCHS = 50                # Number of epochs
MAX_SEQ_LENGTH_MODEL = 1024 # Max sequence length for RoPE precomputation and model capacity
MAX_SEQ_LENGTH_LOADER = 128 # Max length for padding/truncation in DataLoader (<= MAX_SEQ_LENGTH_MODEL)

# Model Hyperparameters (Ensure these match the RoPE version in Bart.py)
D_MODEL = 512
N_HEADS = 8
FFN_DIM = 2048
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 3
DROPOUT = 0.1
ROPE_BASE = 10000 # Base for RoPE calculation

# --- RoPE Implementation ---
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
    if freqs_cos.dim() == 2: # [S, Dh/2]
        if len(x.shape) == 4: # [B, H, S, Dh]
            freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(1)
            freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(1)
        elif len(x.shape) == 3: # [B, S, D]
             freqs_cos = freqs_cos.unsqueeze(0)
             freqs_sin = freqs_sin.unsqueeze(0)
    rotated_x1 = x1 * freqs_cos - x2 * freqs_sin
    rotated_x2 = x1 * freqs_sin + x2 * freqs_cos
    rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)
    return rotated_x.type_as(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: int = 10000):
        super().__init__(); assert dim % 2 == 0; self.dim = dim; self.max_seq_len = max_seq_len; self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)
    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len: seq_len = self.max_seq_len # Simple truncation handling
        return self.freqs_cos[:seq_len, :].to(device), self.freqs_sin[:seq_len, :].to(device)

# --- Model Definition (RoPE Enhanced) ---
def gelu(x): return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BartAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__(); assert d_model % n_heads == 0; self.d_model=d_model; self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.q_proj=nn.Linear(d_model,d_model); self.k_proj=nn.Linear(d_model,d_model); self.v_proj=nn.Linear(d_model,d_model); self.out_proj=nn.Linear(d_model,d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, query, key, value, freqs_cos=None, freqs_sin=None, attn_mask=None, key_padding_mask=None):
        L_q, B, _ = query.size(); L_k = key.size(0); L_v = value.size(0)
        Q = self.q_proj(query).view(L_q, B, self.n_heads, self.head_dim)
        K = self.k_proj(key).view(L_k, B, self.n_heads, self.head_dim)
        V = self.v_proj(value).view(L_v, B, self.n_heads, self.head_dim)
        Q = Q.permute(1, 2, 0, 3); K = K.permute(1, 2, 0, 3) # -> [B, H, L, Dh]
        if freqs_cos is not None and freqs_sin is not None:
            Q = apply_rotary_pos_emb(Q, freqs_cos[:L_q], freqs_sin[:L_q])
            K = apply_rotary_pos_emb(K, freqs_cos[:L_k], freqs_sin[:L_k])
        V = V.permute(1, 2, 0, 3)
        # Combine masks (Boolean OR, True means ignore)
        combined_mask = None
        if key_padding_mask is not None: combined_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L_k]
        if attn_mask is not None:
            bool_attn_mask = attn_mask.bool() if attn_mask.dtype != torch.bool else attn_mask # [L_q, L_k] or similar
            mask_expanded = bool_attn_mask.unsqueeze(0).unsqueeze(1) # -> [1, 1, L_q, L_k]
            combined_mask = combined_mask | mask_expanded if combined_mask is not None else mask_expanded
        # Use PyTorch's optimized attention
        attn_output = nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=combined_mask, dropout_p=self.dropout.p if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.d_model) # [B, L_q, D]
        attn_output = attn_output.transpose(0, 1) # -> [L_q, B, D]
        return self.out_proj(attn_output)

class BartEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__(); self.self_attn=BartAttention(d_model,n_heads,dropout); self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout); self.fc1=nn.Linear(d_model,ffn_dim); self.fc2=nn.Linear(ffn_dim,d_model)
    def forward(self, x, freqs_cos, freqs_sin, src_key_padding_mask=None):
        a=self.self_attn(x, x, x, freqs_cos, freqs_sin, key_padding_mask=src_key_padding_mask); x=self.norm1(x+self.dropout(a))
        f=self.fc2(gelu(self.fc1(x))); x=self.norm2(x+self.dropout(f)); return x

class BartDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__(); self.self_attn=BartAttention(d_model,n_heads,dropout); self.cross_attn=BartAttention(d_model,n_heads,dropout)
        self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model); self.norm3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout); self.fc1=nn.Linear(d_model,ffn_dim); self.fc2=nn.Linear(ffn_dim,d_model)
    def forward(self, x, memory, freqs_cos, freqs_sin, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        s=self.self_attn(x, x, x, freqs_cos, freqs_sin, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask); x=self.norm1(x+self.dropout(s))
        c=self.cross_attn(x, memory, memory, freqs_cos=None, freqs_sin=None, key_padding_mask=memory_key_padding_mask); x=self.norm2(x+self.dropout(c))
        f=self.fc2(gelu(self.fc1(x))); x=self.norm3(x+self.dropout(f)); return x

class BartEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, ffn_dim, num_layers, pad_id, rotary_emb, dropout=0.1):
        super().__init__(); self.embed_tokens=nn.Embedding(vocab_size,d_model,padding_idx=pad_id); self.rotary_emb=rotary_emb
        self.layers=nn.ModuleList([BartEncoderLayer(d_model,n_heads,ffn_dim,dropout) for _ in range(num_layers)])
        self.layer_norm=nn.LayerNorm(d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, src_tokens, src_key_padding_mask=None):
        seq_len=src_tokens.size(1); x=self.dropout(self.embed_tokens(src_tokens)); x=x.transpose(0,1) # [L, B, D]
        f_cos, f_sin = self.rotary_emb(seq_len, x.device)
        for l in self.layers: x = l(x, f_cos, f_sin, src_key_padding_mask)
        x=self.layer_norm(x); return x

class BartDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, ffn_dim, num_layers, pad_id, rotary_emb, dropout=0.1):
        super().__init__(); self.embed_tokens=nn.Embedding(vocab_size,d_model,padding_idx=pad_id); self.rotary_emb=rotary_emb
        self.layers=nn.ModuleList([BartDecoderLayer(d_model,n_heads,ffn_dim,dropout) for _ in range(num_layers)])
        self.layer_norm=nn.LayerNorm(d_model); self.dropout=nn.Dropout(dropout)
    def _generate_square_subsequent_mask(self, sz, device): # Causal mask generation
        mask = torch.triu(torch.full((sz, sz), True, dtype=torch.bool, device=device), diagonal=1)
        return mask # Boolean mask, True means ignore
    def forward(self, tgt_tokens, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        seq_len=tgt_tokens.size(1); x=self.dropout(self.embed_tokens(tgt_tokens)); x=x.transpose(0,1) # [L_tgt, B, D]
        f_cos, f_sin = self.rotary_emb(seq_len, x.device)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, x.device)
        for l in self.layers: x = l(x, memory, f_cos, f_sin, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)
        x=self.layer_norm(x); return x.transpose(0,1) # [B, L_tgt, D]

class BartModel(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim,
                 num_encoder_layers, num_decoder_layers, pad_id, rope_base, dropout=0.1):
        super().__init__(); head_dim = d_model // n_heads; assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE"
        self.rotary_emb=RotaryEmbedding(dim=head_dim, max_seq_len=max_positions, base=rope_base)
        self.encoder=BartEncoder(vocab_size,d_model,n_heads,ffn_dim,num_encoder_layers,pad_id,self.rotary_emb,dropout)
        self.decoder=BartDecoder(vocab_size,d_model,n_heads,ffn_dim,num_decoder_layers,pad_id,self.rotary_emb,dropout)
        self.output_proj=nn.Linear(d_model,vocab_size); self.time_head=nn.Linear(d_model,1)
    def forward(self, src_tokens, tgt_tokens, src_key_padding_mask=None, tgt_key_padding_mask=None):
        mem=self.encoder(src_tokens,src_key_padding_mask=src_key_padding_mask)
        dec=self.decoder(tgt_tokens,mem,memory_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        logits=self.output_proj(dec); time_pred=self.time_head(dec).squeeze(-1); return logits,time_pred

# --- Dataset and DataLoader ---
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

# --- Evaluation Metrics and Function ---
def damerau_levenshtein_similarity(p,t):
    if not isinstance(p,(list,str)): p=tuple(map(str,p)) # Use tuples for rapidfuzz
    else: p = tuple(p)
    if not isinstance(t,(list,str)): t=tuple(map(str,t))
    else: t = tuple(t)
    d=DamerauLevenshtein.distance(p,t); m=max(len(p),len(t)); return 1.0-(d/m) if m>0 else 1.0

def mean_absolute_error(p,t,inv=False):
    pn,tn=np.array(p,dtype=np.float64),np.array(t,dtype=np.float64); v=np.isfinite(pn)&np.isfinite(tn); pf,tf=pn[v],tn[v]
    if len(pf)==0: return 0.0
    if inv: pi,ti=np.expm1(np.maximum(pf,-1e6)),np.expm1(np.maximum(tf,-1e6))
    else: pi,ti=pf,tf
    return np.mean(np.abs(pi-ti))

def evaluate_model(model, dataloader, device, ce_loss_fn, mse_loss_fn, id_to_event, pad_id):
    model.eval(); total_c_loss=0; total_t_loss=0; preds_e=[]; targs_e=[]; preds_d=[]; targs_d=[]; n_samples=0
    print("Running evaluation...");
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            tok=batch['tokens'].to(device); dlt=batch['deltas']; mask=batch['key_padding_mask'].to(device); b_size=tok.size(0); n_samples+=b_size
            t_in=tok[:,:-1]; m_in=mask[:,:-1]; t_out=tok[:,1:]; d_out=dlt[:,1:] # Targets
            logits, t_pred = model(tok,t_in,mask,m_in)
            c_loss=ce_loss_fn(logits.reshape(-1,logits.size(-1)), t_out.reshape(-1).to(device))
            loss_mask=(t_out!=pad_id).reshape(-1).to(device) # Mask for time loss where target is not pad
            t_loss=mse_loss_fn(t_pred.reshape(-1)[loss_mask], d_out.reshape(-1).to(device)[loss_mask]) if loss_mask.any() else torch.tensor(0.0).to(device)
            total_c_loss+=c_loss.item()*b_size; total_t_loss+=t_loss.item()*b_size
            p_ids=torch.argmax(logits,dim=-1).cpu(); p_dlt=t_pred.cpu().numpy(); t_ids=t_out.cpu(); t_dlt=d_out.numpy(); t_mask=mask[:,1:].cpu()
            for i in range(b_size):
                l=(~t_mask[i]).sum().item();
                if l==0: continue
                pi=p_ids[i][:l].tolist(); ti=t_ids[i][:l].tolist(); pd=p_dlt[i][:l]; td=t_dlt[i][:l]
                pe=[id_to_event.get(x,'<unk>') for x in pi]; te=[id_to_event.get(x,'<unk>') for x in ti]
                preds_e.append(pe); targs_e.append(te); preds_d.extend(pd); targs_d.extend(td)
    dls_s=[damerau_levenshtein_similarity(p,t) for p,t in zip(preds_e,targs_e)]; m_dls=np.mean(dls_s) if dls_s else 0.0
    m_mae=mean_absolute_error(preds_d,targs_d,inv=True)/60.0 if preds_d else 0.0
    avg_c_loss=total_c_loss/n_samples if n_samples>0 else 0; avg_t_loss=total_t_loss/n_samples if n_samples>0 else 0
    print(f"\nEval Results: C_Loss {avg_c_loss:.4f}, T_Loss {avg_t_loss:.4f} | DLS {m_dls:.4f}, MAE {m_mae:.4f}")
    model.train(); return m_dls, m_mae

# --- Training Loop Function ---
def run_training(model, train_loader, test_loader, optimizer, ce_loss_fn, mse_loss_fn, device, epochs, id_to_event, pad_id,
                 checkpoint_dir, best_model_path, start_epoch=0, initial_best_dls=-1.0, save_every=10, evaluate_every=1):
    print(f"--- Starting RoPE Model Training (Epochs {start_epoch+1}-{epochs}) ---")
    os.makedirs(checkpoint_dir, exist_ok=True); best_dls = initial_best_dls
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    for epoch in range(start_epoch, epochs):
        model.train(); total_loss=0; total_c_loss=0; total_t_loss=0; num_batches=len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train", leave=False)
        for batch in pbar:
            tok=batch['tokens'].to(device); dlt=batch['deltas'].to(device); mask=batch['key_padding_mask'].to(device)
            t_in=tok[:,:-1]; t_out=tok[:,1:]; d_out=dlt[:,1:]; m_in=mask[:,:-1] # Shifted inputs/outputs
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, t_pred = model(tok,t_in,mask,m_in)
                c_loss=ce_loss_fn(logits.reshape(-1,logits.size(-1)), t_out.reshape(-1))
                loss_mask=(t_out!=pad_id).reshape(-1) # Mask for time loss
                t_loss=mse_loss_fn(t_pred.reshape(-1)[loss_mask], d_out.reshape(-1)[loss_mask]) if loss_mask.any() else torch.tensor(0.0).to(device)
                loss = c_loss + t_loss # Combine loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss+=loss.item(); total_c_loss+=c_loss.item(); total_t_loss+=t_loss.item()
            pbar.set_postfix({'L':f'{loss.item():.3f}','CL':f'{c_loss.item():.3f}','TL':f'{t_loss.item():.3f}'})
        avg_loss=total_loss/num_batches; avg_c=total_c_loss/num_batches; avg_t=total_t_loss/num_batches
        print(f"\nEp {epoch+1} Avg Loss: {avg_loss:.4f} (C:{avg_c:.4f}, T:{avg_t:.4f})")
        if (epoch + 1) % evaluate_every == 0:
            curr_dls, curr_mae = evaluate_model(model, test_loader, device, ce_loss_fn, mse_loss_fn, id_to_event, pad_id)
            if curr_dls > best_dls: best_dls = curr_dls; torch.save(model.state_dict(), best_model_path); print(f"*** Best DLS: {best_dls:.4f}. Saved model. ***")
            else: print(f"DLS {curr_dls:.4f} (Best: {best_dls:.4f})")
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            chk_path=os.path.join(checkpoint_dir, f"rope_bart_epoch_{epoch+1}.pth"); print(f"Saving chkpt: {chk_path}")
            torch.save({'epoch':epoch+1, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),
                        'scaler_state_dict':scaler.state_dict(), 'best_dls':best_dls, 'loss':avg_loss}, chk_path)
    print(f"\n--- Training Complete. Best DLS: {best_dls:.4f} ---"); return best_dls

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(VOCAB_INFO_PATH):
        print(f"Error: Preprocessed data/vocab not found. Run preprocess_data_rope.py first."); exit(1)
    print("Loading preprocessed data..."); data = torch.load(PROCESSED_DATA_PATH); vocab_info = torch.load(VOCAB_INFO_PATH); print("Loaded.")
    train_seqs=data['train_seqs']; train_deltas=data['train_deltas']; test_seqs=data['test_seqs']; test_deltas=data['test_deltas']
    vocab_size=vocab_info['vocab_size']; pad_id=vocab_info['pad_id']; id_to_event=vocab_info['id_to_event']
    print("Creating DataLoaders..."); train_ds=ProcessedSeqDataset(train_seqs,train_deltas); test_ds=ProcessedSeqDataset(test_seqs,test_deltas)
    coll_train=partial(collate_fn,pad_id=pad_id,max_len=MAX_SEQ_LENGTH_LOADER); coll_test=partial(collate_fn,pad_id=pad_id,max_len=MAX_SEQ_LENGTH_LOADER)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=coll_train,num_workers=4,pin_memory=True)
    test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,collate_fn=coll_test,num_workers=4,pin_memory=True); print("DataLoaders created.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    model = BartModel(vocab_size, MAX_SEQ_LENGTH_MODEL, D_MODEL, N_HEADS, FFN_DIM, NUM_ENC_LAYERS, NUM_DEC_LAYERS, pad_id, ROPE_BASE, DROPOUT).to(device)
    print(f"Model instantiated ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params).")
    optimizer=optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=0.01); ce_loss=nn.CrossEntropyLoss(ignore_index=pad_id); mse_loss=nn.MSELoss()
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()) # Define scaler here for checkpoint loading
    start_epoch=0; best_dls=-1.0
    if LOAD_CHECKPOINT_PATH and os.path.isfile(LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint: {LOAD_CHECKPOINT_PATH}"); chkpt=torch.load(LOAD_CHECKPOINT_PATH,map_location=device)
        try: model.load_state_dict(chkpt['model_state_dict']); print("Model state loaded.")
        except RuntimeError as e: print(f"Warn model load: {e}. Trying non-strict."); model.load_state_dict(chkpt['model_state_dict'],strict=False)
        if 'optimizer_state_dict' in chkpt:
            try: optimizer.load_state_dict(chkpt['optimizer_state_dict']); print("Optimizer loaded.")
            except: print("Warn: Optimizer load failed.")
        if 'scaler_state_dict' in chkpt and torch.cuda.is_available(): scaler.load_state_dict(chkpt['scaler_state_dict']); print("Scaler loaded.")
        start_epoch=chkpt.get('epoch',0); best_dls=chkpt.get('best_dls',-1.0); print(f"Resuming ep {start_epoch+1}. Best DLS: {best_dls:.4f}")
    else: print("Starting from scratch.")
    run_training(model,train_loader,test_loader,optimizer,ce_loss,mse_loss,device,EPOCHS,id_to_event,pad_id,
                 CHECKPOINT_DIR,BEST_MODEL_PATH,start_epoch,best_dls,SAVE_CHECKPOINT_EVERY,EVALUATE_EVERY)
    print("--- Training Script Finished ---")