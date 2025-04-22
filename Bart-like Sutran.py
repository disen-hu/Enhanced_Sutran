import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from datetime import timedelta
import os # 用于检查文件是否存在
from tqdm import tqdm
# 注意：确保你安装了 rapidfuzz: pip install rapidfuzz
from rapidfuzz.distance import DamerauLevenshtein
# from nltk.metrics.distance import edit_distance # 如果不用 NLTK 可以注释掉

############################
# 1. 数据处理: 增加时间特征
############################

# --- 数据加载和预处理 (与你原代码相同) ---
try:
    df = pd.read_csv('BPIC_19_train_initial.csv')
    test_df = pd.read_csv('BPIC_19_test_initial.csv')
except FileNotFoundError as e:
    print(f"错误：找不到数据文件 {e.filename}。请确保文件在正确的路径下。")
    exit()


df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
test_df['time:timestamp'] = pd.to_datetime(test_df['time:timestamp'])

# --- 训练集处理 ---
grouped = df.groupby('case:concept:name', group_keys=True)
all_sequences = []
all_time_deltas = []
for case_id, group in grouped:
    group = group.sort_values('time:timestamp')
    events = group['concept:name'].tolist()
    times = group['time:timestamp'].tolist()
    deltas = [0.0]
    for i in range(1, len(times)):
        delta_sec = (times[i] - times[i-1]).total_seconds()
        deltas.append(delta_sec)
    deltas = np.log1p(deltas)
    all_sequences.append(events)
    all_time_deltas.append(deltas)

# --- 构建词表 ---
all_events = set()
for seq in all_sequences:
    for e in seq:
        all_events.add(e)

special_tokens = ['<pad>', '<s>', '</s>', '<mask>']
vocab = special_tokens + sorted(all_events)
event_to_id = {w: i for i, w in enumerate(vocab)}
id_to_event = {i: w for w, i in event_to_id.items()}

pad_id = event_to_id['<pad>']
s_id = event_to_id['<s>']
eos_id = event_to_id['</s>']
mask_id = event_to_id['<mask>']
vocab_size = len(vocab)

# --- 编码训练序列 ---
encoded_sequences = []
encoded_deltas = []
for seq, dts in zip(all_sequences, all_time_deltas):
    tokens = [s_id] + [event_to_id[e] for e in seq] + [eos_id]
    dts = [0.0] + list(dts) + [0.0]
    if len(dts) < len(tokens):
        dts += [0.0]*(len(tokens)-len(dts))
    encoded_sequences.append(tokens)
    encoded_deltas.append(dts)

# --- 测试集处理 ---
test_grouped = test_df.groupby('case:concept:name', group_keys=True)
test_sequences = []
test_time_deltas = []
for case_id, group in test_grouped:
    group = group.sort_values('time:timestamp')
    events = group['concept:name'].tolist()
    times = group['time:timestamp'].tolist()
    deltas = [0.0]
    for i in range(1, len(times)):
        delta_sec = (times[i] - times[i-1]).total_seconds()
        deltas.append(delta_sec)
    deltas = np.log1p(deltas)
    test_sequences.append(events)
    test_time_deltas.append(deltas)

# --- 编码测试序列 ---
encoded_test_sequences = []
encoded_test_deltas = []
for seq, dts in zip(test_sequences, test_time_deltas):
    # 使用 .get(e, mask_id) 处理测试集中可能出现的未知事件
    tokens = [s_id] + [event_to_id.get(e, mask_id) for e in seq] + [eos_id]
    dts = [0.0] + list(dts) + [0.0]
    if len(dts) < len(tokens):
        dts += [0.0] * (len(tokens) - len(dts))
    encoded_test_sequences.append(tokens)
    encoded_test_deltas.append(dts)


############################
# 2. Dataset和DataLoader
############################
class SeqWithTimeDataset(Dataset):
    def __init__(self, sequences, deltas, pad_id, max_length=50): # 设定一个最大长度，防止内存问题
        self.sequences = sequences
        self.deltas = deltas
        self.pad_id = pad_id
        self.max_length = max_length # 限制序列最大长度

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_length] # 截断
        dts = self.deltas[idx][:self.max_length]   # 截断
        # 返回截断后的数据
        return seq, dts

def collate_fn(batch):
    # batch: [(seq, dts), ...]
    # 先获取当前 batch 中最长的序列长度（截断后）
    max_len_in_batch = max(len(x[0]) for x in batch)

    padded_seq = []
    padded_dts = []
    for seq,dts in batch:
        # 计算需要填充的长度
        seq_padding_len = max_len_in_batch - len(seq)
        dts_padding_len = max_len_in_batch - len(dts)

        # 填充
        seq_pad = seq + [pad_id] * seq_padding_len
        # dts 是浮点数列表，用 0.0 填充
        dts_pad = list(dts) + [0.0] * dts_padding_len

        padded_seq.append(seq_pad)
        padded_dts.append(dts_pad)

    padded_seq = torch.tensor(padded_seq, dtype=torch.long)
    padded_dts = torch.tensor(padded_dts, dtype=torch.float)
    # key_padding_mask 应该标记 PAD 的位置为 True
    key_padding_mask = (padded_seq == pad_id)
    return {
        'tokens': padded_seq,
        'deltas': padded_dts,
        'key_padding_mask': key_padding_mask
    }

# 使用截断后的 Dataset
dataset = SeqWithTimeDataset(encoded_sequences, encoded_deltas, pad_id, max_length=100) # 例如最大长度100
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

test_dataset = SeqWithTimeDataset(encoded_test_sequences, encoded_test_deltas, pad_id, max_length=100) # 同样截断
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


############################
# 3. 模型定义(与之前类似,增加time_head)
############################

# --- 模型定义 (与你原代码相同) ---
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, d_model, pad_idx):
        super().__init__()
        # Embedding 层的大小应该是 max_position_embeddings
        self.embed = nn.Embedding(max_position_embeddings, d_model, padding_idx=pad_idx)
        # 注意：如果你想让 PAD 位置的 positional embedding 为0，可以不设置 padding_idx
        # 或者在 forward 中处理。BART 原始实现通常不将 pad_idx 用于位置嵌入
        # 但在这里，如果词嵌入用了 pad_idx=0，位置嵌入也用可能导致冲突，如果 pad_id=0
        # 推荐位置嵌入不使用 padding_idx，或者使用不同的 pad_id
        # 这里暂时保持你的设置，但需注意 pad_id 不能是0，否则会和第一个位置冲突
        # 如果 pad_id=0, 建议改为：
        # self.embed = nn.Embedding(max_position_embeddings, d_model)
        self.pad_idx = pad_idx # 保存以备后用（如果需要的话）

    def forward(self, x):
        # x 的形状是 [B, L]
        B, L = x.size()
        # 创建位置索引 [0, 1, ..., L-1]
        positions = torch.arange(L, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, L)
        # 获取位置嵌入
        position_embeddings = self.embed(positions)
        return position_embeddings

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BartAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # 输入形状: [L, B, D] (PyTorch Transformer 默认)
        L_q, B, _ = query.size()
        L_k = key.size(0)

        # 1. 线性投射 [L, B, D] -> [L, B, D]
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 2. 拆分头 & 调整维度 [L, B, D] -> [L, B, nH, hD] -> [B, nH, L, hD]
        Q = Q.view(L_q, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3) # [B, nH, L_q, hD]
        K = K.view(L_k, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3) # [B, nH, L_k, hD]
        V = V.view(L_k, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3) # [B, nH, L_k, hD]

        # 3. 计算注意力分数 [B, nH, L_q, hD] @ [B, nH, hD, L_k] -> [B, nH, L_q, L_k]
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 应用 key_padding_mask (标记 PAD token)
        # key_padding_mask: [B, L_k], 需要扩展到 [B, 1, 1, L_k]
        if key_padding_mask is not None:
            # 我们需要 mask 掉 key 中的 PAD token 对所有 query 的注意力
            # mask 应该在 L_k 维度上生效
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L_k]
            attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))

        # 5. 应用 attn_mask (例如 decoder 的 causal mask)
        # attn_mask: 通常是 [L_q, L_k] 或 [B, nH, L_q, L_k]
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask # 加法，因为 mask 通常是 0 和 -inf

        # 6. Softmax 获取概率
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 7. 加权 V [B, nH, L_q, L_k] @ [B, nH, L_k, hD] -> [B, nH, L_q, hD]
        attn_output = torch.matmul(attn_probs, V)

        # 8. 合并头 & 调整维度 [B, nH, L_q, hD] -> [B, L_q, nH, hD] -> [B, L_q, D]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(B, L_q, self.d_model)

        # 9. 转换回 PyTorch Transformer 的 [L, B, D] 格式
        attn_output = attn_output.transpose(0, 1) # [L_q, B, D]

        # 10. 输出线性层
        attn_output = self.out_proj(attn_output)

        return attn_output # 输出形状 [L_q, B, D]

class BartEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = BartAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # Dropout 通常加在残差连接之前
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.activation = gelu # 使用 gelu 激活函数

    def forward(self, x, src_key_padding_mask=None):
        # x shape: [L, B, D]
        # src_key_padding_mask: [B, L]

        # 1. Self-Attention Block
        residual = x
        attn_output = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=src_key_padding_mask # 正确传递 mask
        )
        x = self.norm1(residual + self.dropout(attn_output)) # Add & Norm

        # 2. Feed-Forward Block
        residual = x
        ff_output = self.fc2(self.dropout(self.activation(self.fc1(x)))) # FFN
        x = self.norm2(residual + self.dropout(ff_output)) # Add & Norm

        return x # shape: [L, B, D]

class BartDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = BartAttention(d_model, n_heads, dropout)
        self.cross_attn = BartAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.activation = gelu

    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # x shape: [L_tgt, B, D]
        # memory shape: [L_src, B, D]
        # tgt_mask: [L_tgt, L_tgt] (causal mask)
        # memory_key_padding_mask: [B, L_src] (from encoder)
        # tgt_key_padding_mask: [B, L_tgt] (for target sequence)

        # 1. Self-Attention Block (with causal mask and target padding mask)
        residual = x
        self_attn_output = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=tgt_mask, # Causal mask
            key_padding_mask=tgt_key_padding_mask # Mask padding in target
        )
        x = self.norm1(residual + self.dropout(self_attn_output))

        # 2. Cross-Attention Block (attend to encoder memory)
        residual = x
        cross_attn_output = self.cross_attn(
            query=x, # Query from decoder
            key=memory, # Key from encoder memory
            value=memory, # Value from encoder memory
            key_padding_mask=memory_key_padding_mask # Mask padding in encoder memory
        )
        x = self.norm2(residual + self.dropout(cross_attn_output))

        # 3. Feed-Forward Block
        residual = x
        ff_output = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = self.norm3(residual + self.dropout(ff_output))

        return x # shape: [L_tgt, B, D]

class BartEncoder(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim, num_layers, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # 确保 LearnedPositionalEmbedding 的 max_position_embeddings 参数正确
        # 通常需要比最大序列长度稍大，例如 max_length + 2 (因为加了<s></s>)
        # 或者直接使用一个较大的固定值如 512, 1024
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, pad_id)
        self.layers = nn.ModuleList([BartEncoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model) # Final layer norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens, src_key_padding_mask=None):
        # src_tokens: [B, L]
        # src_key_padding_mask: [B, L]

        # 1. Embeddings
        token_embed = self.embed_tokens(src_tokens) # [B, L, D]
        pos_embed = self.embed_positions(src_tokens) # [B, L, D]
        x = token_embed + pos_embed
        x = self.dropout(x)

        # 2. Transpose for Transformer layers: [B, L, D] -> [L, B, D]
        x = x.transpose(0, 1)

        # 3. Pass through Encoder Layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        # 4. Final Layer Norm (optional, BART does this)
        x = self.layer_norm(x)

        # Output shape: [L, B, D] (memory for the decoder)
        return x

class BartDecoder(nn.Module):
    def __init__(self, vocab_size, max_positions, d_model, n_heads, ffn_dim, num_layers, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, pad_id)
        self.layers = nn.ModuleList([BartDecoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model) # Final layer norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_tokens, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt_tokens: [B, L_tgt]
        # memory: [L_src, B, D] (from encoder)
        # tgt_key_padding_mask: [B, L_tgt]
        # memory_key_padding_mask: [B, L_src]

        # 1. Embeddings
        token_embed = self.embed_tokens(tgt_tokens) # [B, L_tgt, D]
        pos_embed = self.embed_positions(tgt_tokens) # [B, L_tgt, D]
        x = token_embed + pos_embed
        x = self.dropout(x)

        # 2. Transpose for Transformer layers: [B, L_tgt, D] -> [L_tgt, B, D]
        x = x.transpose(0, 1)

        # 3. Generate Causal Mask
        L_tgt = x.size(0)
        tgt_mask = self._generate_square_subsequent_mask(L_tgt, x.device)

        # 4. Pass through Decoder Layers
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        # 5. Final Layer Norm
        x = self.layer_norm(x)

        # 6. Transpose back to [B, L_tgt, D] for output projection
        x = x.transpose(0, 1)

        return x # Shape: [B, L_tgt, D]

    def _generate_square_subsequent_mask(self, sz, device):
        # Creates a mask like:
        # [[0., -inf, -inf],
        #  [0., 0., -inf],
        #  [0., 0., 0.]]
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class BartModel(nn.Module):
    # 增加 max_positions 参数
    def __init__(self, vocab_size, max_positions=102, d_model=768, n_heads=12, ffn_dim=3072,
                 num_encoder_layers=6, num_decoder_layers=6, pad_id=0, dropout=0.1):
        super().__init__()
        # 确保传递 max_positions 给 Encoder 和 Decoder
        self.encoder = BartEncoder(vocab_size, max_positions, d_model, n_heads, ffn_dim, num_encoder_layers, pad_id, dropout)
        self.decoder = BartDecoder(vocab_size, max_positions, d_model, n_heads, ffn_dim, num_decoder_layers, pad_id, dropout)

        # 输出层权重可以与输入嵌入层共享（可选，BART 常用技巧）
        # self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # self.output_proj.weight = self.encoder.embed_tokens.weight # 权重共享

        # 如果不共享权重，则单独定义输出层
        self.output_proj = nn.Linear(d_model, vocab_size)

        # 增加时间预测头
        self.time_head = nn.Linear(d_model, 1)

    def forward(self, src_tokens, tgt_tokens, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src_tokens: [B, L_src]
        # tgt_tokens: [B, L_tgt]
        # src_key_padding_mask: [B, L_src]
        # tgt_key_padding_mask: [B, L_tgt]

        # Encoder forward pass
        # memory shape: [L_src, B, D]
        memory = self.encoder(src_tokens, src_key_padding_mask=src_key_padding_mask)

        # Decoder forward pass
        # dec_out shape: [B, L_tgt, D]
        dec_out = self.decoder(
            tgt_tokens,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Encoder的padding mask 用于 cross-attention
        )

        # Output projection for vocabulary logits
        logits = self.output_proj(dec_out)  # [B, L_tgt, vocab_size]

        # Time prediction head
        time_pred = self.time_head(dec_out).squeeze(-1) # [B, L_tgt]

        return logits, time_pred

############################
# 评估函数定义
############################

# 使用 rapidfuzz 计算 Damerau-Levenshtein 相似度
def damerau_levenshtein_similarity(pred_seq, target_seq):
    """计算 Damerau-Levenshtein 相似度"""
    # rapidfuzz 需要字符串或列表作为输入
    dl_distance = DamerauLevenshtein.distance(pred_seq, target_seq)
    max_len = max(len(pred_seq), len(target_seq))
    return 1.0 - dl_distance / max_len if max_len > 0 else 1.0

# Mean Absolute Error (MAE) 函数
def mean_absolute_error(pred, target, inverse_transform=False):
    """计算平均绝对误差，并支持 log(1 + delta) 的逆向变换"""
    pred = np.array(pred)
    target = np.array(target)

    # 过滤掉 NaN 或 Inf 值（可能由 log1p 或 expm1 产生）
    valid_indices = np.isfinite(pred) & np.isfinite(target)
    pred = pred[valid_indices]
    target = target[valid_indices]

    if len(pred) == 0: # 如果没有有效数据点
        return 0.0

    # 如果需要逆向变换 (从 log(1+delta) 转换回 delta)
    if inverse_transform:
        # 确保在逆变换前值是合理的，避免 overflow
        # log1p 的结果通常 >= 0，所以逆变换输入也应 >= 0
        pred = np.expm1(np.maximum(pred, 0)) # e^(log_delta) - 1
        target = np.expm1(np.maximum(target, 0))

    return np.mean(np.abs(pred - target))


############################
# 训练设置和循环
############################

# --- 参数设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型参数 (根据你的 BART 实现调整)
# 注意: max_positions 应 >= dataloader 中的 max_length
model_args = {
    'vocab_size': vocab_size,
    'max_positions': 102, # 例如 100(data) + 2(special tokens)
    'd_model': 768,       # BART-base 常用 768
    'n_heads': 12,        # BART-base 常用 12
    'ffn_dim': 3072,      # BART-base 常用 3072 (4 * d_model)
    'num_encoder_layers': 6, # BART-base 常用 6
    'num_decoder_layers': 6, # BART-base 常用 6
    'pad_id': pad_id,
    'dropout': 0.1,
}
model = BartModel(**model_args).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5) # 可以调整学习率

# --- 检查点加载 ---
# <<< 修改开始 >>>
checkpoint_to_load_path = "model_epoch_300.pth"  # <--- ***修改这里*** 指向你要加载的检查点文件
start_epoch = 0
best_dls = -1 # 初始化最佳 DLS

if checkpoint_to_load_path and os.path.isfile(checkpoint_to_load_path):
    print(f"Attempting to load checkpoint: {checkpoint_to_load_path}")
    try:
        checkpoint = torch.load(checkpoint_to_load_path, map_location=device)

        # 检查 checkpoint 类型
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 新格式：包含字典
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded successfully.")
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully.")
            else:
                 print("Warning: Optimizer state not found in checkpoint.")
            # 从检查点获取 epoch，如果没有则默认为0，并加1作为起始
            start_epoch = checkpoint.get('epoch', 0)
            # 尝试从检查点恢复最佳 DLS (如果保存了的话)
            best_dls = checkpoint.get('best_dls', -1)
            print(f"Resuming training from epoch {start_epoch + 1}")

        else:
            # 旧格式：只包含 model state_dict
            model.load_state_dict(checkpoint)
            start_epoch = 0 # 无法从旧格式获取 epoch，从头开始或手动指定
            print("Model state loaded successfully (old format).")
            print("Warning: Optimizer state not loaded (old format checkpoint).")
            print(f"Warning: Could not determine epoch from checkpoint, starting from epoch 1 (index 0). You might need to adjust the starting epoch.")
            # 注意：如果加载的是旧的 best_model.pth，它可能是在某个 epoch 结束时保存的
            # 你可能需要根据文件名或其他记录来确定应该从哪个 epoch 继续

        print(f"Checkpoint loaded. Resuming/Starting training from epoch index {start_epoch}.")

    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting training from scratch.")
        start_epoch = 0
        best_dls = -1
else:
    print("No checkpoint found or specified. Starting training from scratch.")
    start_epoch = 0
    best_dls = -1
# <<< 修改结束 >>>


# --- 损失函数 ---
# 定义交叉熵损失函数，忽略 pad_id
# 确保 pad_id 在这里是正确的整数索引
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
mse_loss_fn = nn.MSELoss() # 定义均方误差损失函数


# --- 训练循环 ---
epochs = 1000 # 总共希望训练到的 epoch 数
print(f"Starting training loop from epoch {start_epoch + 1} up to {epochs}")

for epoch in range(start_epoch, epochs):
    model.train() # 设置为训练模式
    total_loss = 0
    total_concept_loss = 0
    total_time_loss = 0

    # 使用 tqdm 显示进度条
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for batch in progress_bar:
        src_tokens = batch['tokens'].to(device) # [B, L]
        tgt_tokens = batch['tokens'].to(device) # [B, L] 通常 target 和 source 相同或有 slight shift
        deltas = batch['deltas'].to(device) # [B, L]
        src_key_padding_mask = batch['key_padding_mask'].to(device) # [B, L]
        tgt_key_padding_mask = batch['key_padding_mask'].to(device) # [B, L]

        # 模型前向传播
        logits, time_pred = model(
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens[:, :-1], # Target input 通常是去掉最后一个 token
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask[:, :-1] # Target mask 也对应去掉最后一个
        )
        # logits shape: [B, L-1, vocab_size]
        # time_pred shape: [B, L-1]

        # 计算损失
        # 1. Concept Loss (预测下一个事件)
        # Logits 需要 reshape: [B, L-1, V] -> [B*(L-1), V]
        # Labels 需要是对应的 target: [B, L], 取 [:, 1:] -> [B, L-1] -> [B*(L-1)]
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = tgt_tokens[:, 1:].contiguous().view(-1)
        concept_loss = ce_loss_fn(shift_logits, shift_labels)

        # 2. Time Loss (预测下一个时间差)
        # Predictions: time_pred [B, L-1]
        # Labels: deltas [B, L], 取 [:, 1:] -> [B, L-1]
        # 注意：确保 time_pred 和 target deltas 对应正确
        shift_deltas = deltas[:, 1:].contiguous() # Target 时间差
        time_loss = mse_loss_fn(time_pred.contiguous(), shift_deltas)

        # --- 合并损失 (可以加权重) ---
        loss = concept_loss + time_loss # 这里是简单相加，可以调整权重 alpha * concept + beta * time

        # --- 反向传播和优化 ---
        optimizer.zero_grad()
        loss.backward()
        # 可以添加梯度裁剪防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_concept_loss += concept_loss.item()
        total_time_loss += time_loss.item()

        # 更新 tqdm 进度条显示的信息
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'ConceptL': f'{concept_loss.item():.4f}',
            'TimeL': f'{time_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_concept_loss = total_concept_loss / len(dataloader)
    avg_time_loss = total_time_loss / len(dataloader)
    print(f"\nEpoch {epoch+1}/{epochs} finished.")
    print(f"  Avg Loss: {avg_loss:.4f} | Avg Concept Loss: {avg_concept_loss:.4f} | Avg Time Loss: {avg_time_loss:.4f}")


    # --- 模型评估 (例如每隔 N 个 epoch 评估一次) ---
    if (epoch + 1) % 1 == 0: # 每个 epoch 都评估
        model.eval() # 设置为评估模式
        dls_scores = []
        timestamp_maes = []
        # runtime_maes = [] # 如果需要运行时MAE，取消注释

        print("Running evaluation on test set...")
        with torch.no_grad(): # 评估时不需要计算梯度
            for batch in tqdm(test_dataloader, desc="Evaluating", leave=False):
                tokens = batch['tokens'].to(device) #[B, L]
                deltas = batch['deltas'] # Target deltas [B, L], 保留在 CPU 上用于比较
                key_padding_mask = batch['key_padding_mask'].to(device) #[B, L]

                # 在评估时，通常也提供目标序列作为输入，让模型预测整个序列
                # 注意：输入给 decoder 的 target 需要去掉最后一个 token
                eval_tgt_tokens = tokens[:, :-1]
                eval_tgt_mask = key_padding_mask[:, :-1]

                logits, time_pred = model(
                    src_tokens=tokens,
                    tgt_tokens=eval_tgt_tokens,
                    src_key_padding_mask=key_padding_mask,
                    tgt_key_padding_mask=eval_tgt_mask
                 )
                # logits: [B, L-1, V], time_pred: [B, L-1]

                # 获取预测的 token id
                pred_token_ids = torch.argmax(logits, dim=-1).cpu() # [B, L-1]
                pred_deltas = time_pred.cpu().numpy() # [B, L-1]

                # 与 Target 对比 (Target 需要移位)
                target_token_ids = tokens[:, 1:].cpu() # [B, L-1]
                target_deltas = deltas[:, 1:].numpy()  # [B, L-1]
                target_padding_mask = key_padding_mask[:, 1:].cpu() # [B, L-1], 对应 target 的 mask

                # 逐个样本计算指标
                for i in range(tokens.size(0)): # 遍历 batch 中的每个样本
                    # 过滤掉 PAD token (使用 target 的 mask)
                    actual_len = (~target_padding_mask[i]).sum().item() # 计算非 PAD 的实际长度
                    if actual_len == 0: continue # 跳过完全是 PAD 的样本

                    pred_ids_seq = pred_token_ids[i][:actual_len].tolist()
                    target_ids_seq = target_token_ids[i][:actual_len].tolist()

                    pred_delta_seq = pred_deltas[i][:actual_len]
                    target_delta_seq = target_deltas[i][:actual_len]

                    # 将 ID 转换回事件名称字符串，用于 DLS 计算
                    # 注意：这里不应该过滤 s_id, eos_id，因为它们是序列的一部分
                    # 但要确保 id_to_event 字典处理所有可能的 ID
                    pred_event_seq = [id_to_event.get(idx, '<unk>') for idx in pred_ids_seq]
                    target_event_seq = [id_to_event.get(idx, '<unk>') for idx in target_ids_seq]

                    # 1. Damerau-Levenshtein Similarity
                    dls = damerau_levenshtein_similarity(pred_event_seq, target_event_seq)
                    dls_scores.append(dls)

                    # 2. Timestamp MAE (预测的下一个时间差的 MAE)
                    # 需要进行逆变换（如果训练时用了 log1p）并转换单位（秒->分钟）
                    ts_mae = mean_absolute_error(pred_delta_seq, target_delta_seq, inverse_transform=True) / 60.0
                    timestamp_maes.append(ts_mae)

                    # 3. Runtime MAE (可选，整个序列累积时间的 MAE)
                    # pred_runtime = np.cumsum(np.expm1(np.maximum(pred_delta_seq, 0))) / 60.0
                    # target_runtime = np.cumsum(np.expm1(np.maximum(target_delta_seq, 0))) / 60.0
                    # # 计算最后一个时间点的 MAE 或者整个序列的平均 MAE
                    # if len(pred_runtime) > 0:
                    #     rt_mae = np.abs(pred_runtime[-1] - target_runtime[-1])
                    #     runtime_maes.append(rt_mae)

        # 计算平均指标
        mean_dls = np.mean(dls_scores) if dls_scores else 0.0
        mean_timestamp_mae = np.mean(timestamp_maes) if timestamp_maes else 0.0
        # mean_runtime_mae = np.mean(runtime_maes) if runtime_maes else 0.0 # 如果计算了 runtime MAE

        print(f"Epoch {epoch+1} Evaluation Results:")
        print(f"  DLS (Similarity): {mean_dls:.4f}")
        print(f"  MAE (Next Timestamp, minutes): {mean_timestamp_mae:.4f}")
        # print(f"  MAE (Total Runtime, minutes): {mean_runtime_mae:.4f}") # 如果计算了 runtime MAE

        # --- 保存最佳模型 (基于 DLS) ---
        if mean_dls > best_dls:
            best_dls = mean_dls
            best_model_save_path = "best_model.pth"
            torch.save(model.state_dict(), best_model_save_path) # 只保存模型权重
            print(f"*** New best model saved to {best_model_save_path} at epoch {epoch+1} with DLS: {best_dls:.4f} ***")

        # 切换回训练模式
        model.train()

    # --- 定期保存检查点 (包含优化器状态和 epoch) ---
    if (epoch + 1) % 10 == 0: # 每 10 个 epoch 保存一次
        checkpoint_save_path = f"model_epoch_{epoch+1}.pth"
        save_dict = {
            'epoch': epoch + 1, # 保存完成的 epoch 编号
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss, # 保存当前 epoch 的平均 loss
            'best_dls': best_dls # 保存当前的最佳 DLS 分数
        }
        torch.save(save_dict, checkpoint_save_path)
        print(f"Checkpoint saved to {checkpoint_save_path} at epoch {epoch+1}")


print(f"\nTraining complete. Best model saved at 'best_model.pth' with DLS: {best_dls:.4f}")