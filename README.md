# Enhanced-SUTRAN: Event Sequence Prediction

## Overview

### main enhancement


## SuTraN_RoPE/SuTraN_RoPE.py
  - New params: decoding_strategy (default "greedy" ), top_p , temperature .
  - Implements nucleus sampling in the decoding branch of forward when decoding_strategy == "nucleus" .
  - Instantiates RoPE-enabled layers ( EncoderLayerRoPE , DecoderLayerRoPE ).
## SuTraN_RoPE/layers_rope.py
  - RoPE core: apply_rope(q, k) ; used inside attention.
  - Attention modules: MultiHeadAttentionRoPE , MultiHeadSelfAttentionDecoderRoPE apply RoPE in forward .
## SuTraN_RoPE/transformer_prefix_encoder_rope.py
  - EncoderLayerRoPE uses MultiHeadAttentionRoPE (RoPE applied in self-attention).
## SuTraN_RoPE/transformer_suffix_decoder_rope.py
  - DecoderLayerRoPE uses RoPE in both self-attention and cross-attention.

## Enable nucleus sampling:
  - decoding_strategy="nucleus" , set top_p = 0.3 for example 

