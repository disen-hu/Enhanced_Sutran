import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_cos_sin(seq_len: int, dim: int, base: float = 10000.0, device: torch.device = device):
    """Compute RoPE cos/sin tensors for given sequence length and head dimension.

    Returns broadcastable cos and sin with shape (1, 1, seq_len, dim).
    """
    assert dim % 2 == 0, "RoPE dimension must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))  # (dim/2,)
    t = torch.arange(seq_len, device=device).float()  # (seq_len,)
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim/2)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (seq_len, dim)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # (seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    return cos, sin

def _rotate_half(x: torch.Tensor) -> torch.Tensor:

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
  
    rotated = torch.stack((-x_odd, x_even), dim=-1)  
    return rotated.reshape(x.shape)

def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # """Apply rotary position embedding to q and k.

    seq_len = q.shape[2]
    dim = q.shape[-1]
    cos, sin = _get_cos_sin(seq_len, dim, device=q.device)
    q_rope = (q * cos) + (_rotate_half(q) * sin)
    k_rope = (k * cos) + (_rotate_half(k) * sin)
    if not getattr(apply_rope, "_logged", False):
        print(f"[RoPE] apply_rope: q={tuple(q.shape)}, k={tuple(k.shape)}, device={q.device}")
        apply_rope._logged = True
    return q_rope, k_rope

class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionRoPE, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert self.d_k % 2 == 0, "head dim must be even for RoPE"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, W, W)
        if mask is not None:
            batch_size = Q.shape[0]
            window_size = Q.shape[2]
            mask = torch.broadcast_to(mask.unsqueeze(1), size=(batch_size, window_size, window_size))
            attn_scores = attn_scores.masked_fill(mask=mask.unsqueeze(1), value=-1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Apply RoPE to Q and K
        Q, K = apply_rope(Q, K)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class MultiHeadSelfAttentionDecoderRoPE(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttentionDecoderRoPE, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert self.d_k % 2 == 0, "head dim must be even for RoPE"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        window_size = Q.shape[2]
        look_ahead = torch.triu(torch.ones(1, 1, window_size, window_size), diagonal=1).bool().to(Q.device)
        attn_scores = attn_scores.masked_fill(mask=look_ahead, value=-1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Apply RoPE to Q and K
        Q, K = apply_rope(Q, K)

        attn_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))