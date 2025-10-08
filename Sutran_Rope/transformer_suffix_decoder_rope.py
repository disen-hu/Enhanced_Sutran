import torch
import torch.nn as nn
from SuTraN_RoPE.layers_rope import MultiHeadAttentionRoPE, PositionWiseFeedForward, MultiHeadSelfAttentionDecoderRoPE


class DecoderLayerRoPE(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayerRoPE, self).__init__()
        
        self.self_attn = MultiHeadSelfAttentionDecoderRoPE(d_model, num_heads)

        self.cross_attn = MultiHeadAttentionRoPE(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x