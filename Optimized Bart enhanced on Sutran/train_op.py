# train_bart_sutran_data_v3_modified_for_tuples.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset # Changed Dataset import
import numpy as np
import os
from tqdm import tqdm
import math
from functools import partial
from rapidfuzz.distance import DamerauLevenshtein
import random
import time
import pickle


METADATA_PATH = 'BPIC_19/metadata_sutran_v2.pt' # 你的元数据文件路径
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

print("Means:", metadata['train_means'])
print("Stds:", metadata['train_stds'])
# 检查是否有接近 0 的 std
near_zero_stds = {k: v for k, v in metadata['train_stds'].items() if np.abs(v) < 1e-6}
if near_zero_stds:
    print("\nWarning: Found near-zero standard deviations:", near_zero_stds)
# Removed default_collate as we now use TensorDataset
# from torch.utils.data.dataloader import default_collate
# --- Constants ---  <---- 添加这个部分
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
END_TOKEN = '<end>' #
# --- Configuration ---
# !!! UPDATE THESE PATHS to match the output of your modified preprocessing !!!
PROCESSED_TENSOR_PATH = 'BPIC_19/processed_tensors_sutran_v2.pt' # Path to saved tensors tuple
METADATA_PATH = 'BPIC_19/metadata_sutran_v2.pt'             # Path to saved metadata
CHECKPOINT_DIR = "bart_sutran_checkpoints_v3_tuples"    # New dir name
BEST_MODEL_PATH = "bart_sutran_best_model_v3_tuples.pth" # New model name
LOAD_CHECKPOINT_PATH = None
SAVE_CHECKPOINT_EVERY = 5
EVALUATE_EVERY = 1
# LOG_TRANSFORM_APPLIED = False # Get this info from loaded metadata if needed
USE_AMP = torch.cuda.is_available()

# Training Hyperparameters (Keep or adjust)
BATCH_SIZE = 512
LEARNING_RATE = 3e-5
EPOCHS = 50
GRADIENT_CLIP_NORM = 1.0

# Model Hyperparameters (Keep or adjust)
D_MODEL = 32 # Match author's d_model for SuTraN
N_HEADS = 8 # Match author's n_heads
FFN_DIM = 4 * D_MODEL # Match author's d_ff
NUM_ENCODER_LAYERS = 4 # Match author's num_prefix_encoder_layers
NUM_DECODER_LAYERS = 4 # Match author's num_decoder_layers
DROPOUT = 0.2 # Match author's dropout

# --- Model Definition (Adapted BART with Learned Positional Embeddings) ---
# Keep LearnedPositionalEmbedding as is
class LearnedPositionalEmbedding(nn.Module):
    """Learned Positional Embeddings."""
    def __init__(self, max_position_embeddings, d_model, pad_idx):
        super().__init__()
        # Pad index is 0 for activity/categoricals in the new data format
        self.embed = nn.Embedding(max_position_embeddings, d_model, padding_idx=pad_idx)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if pad_idx is not None:
             # Ensure pad embedding is 0 only if pad_idx is valid
             if pad_idx >= 0 and pad_idx < max_position_embeddings:
                  nn.init.constant_(self.embed.weight[pad_idx], 0)
             else: print(f"Warning: pad_idx {pad_idx} invalid for embedding size {max_position_embeddings}")


    def forward(self, input_ids_shape, past_key_values_length: int = 0):
        batch_size, seq_length = input_ids_shape
        device = self.embed.weight.device
        positions = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        # Ensure positions don't exceed embedding size
        positions = positions % self.embed.num_embeddings
        return self.embed(positions.expand(batch_size, -1))


# Using standard PyTorch Transformer components
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

# --- Main Adapted BART Model ---
class AdaptedBartModel(nn.Module):
    """
    Combines feature processing with BART Encoder/Decoder using Learned Positional Embeddings.
    MODIFIED to accept a tuple of tensors as input.
    """
    def __init__(self, metadata, d_model, n_heads, ffn_dim, num_enc_layers, num_dec_layers, dropout):
        super().__init__()
        self.metadata = metadata
        self.d_model = d_model
        self.pad_id = metadata['pad_id'] # Should be 0
        self.activity_col = metadata['activity_col']
        # Get feature lists from metadata saved by modified preprocessing
        self.prefix_cat_cols = metadata.get('prefix_cat_cols', [self.activity_col]) # Includes activity
        self.prefix_num_cols = metadata.get('prefix_num_cols', [])
        self.suffix_cat_cols = metadata.get('suffix_cat_cols', [self.activity_col]) # Includes activity
        self.suffix_num_cols = metadata.get('suffix_num_cols', [])
        self.time_label_features = metadata.get('time_label_features', [])
        self.padding_length = metadata['padding_length']

        # --- Determine Feature Indices in Input Tuple ---
        # This needs to EXACTLY match the order in the tensor tuple saved by processdata...
        self.num_prefix_cat_tensors = len(self.prefix_cat_cols)
        self.prefix_num_tensor_idx = self.num_prefix_cat_tensors
        self.prefix_padding_mask_idx = self.prefix_num_tensor_idx + 1
        self.num_suffix_cat_tensors = len(self.suffix_cat_cols)
        self.suffix_cat_tensor_start_idx = self.prefix_padding_mask_idx + 1
        self.suffix_num_tensor_idx = self.suffix_cat_tensor_start_idx + self.num_suffix_cat_tensors
        # Assuming labels follow suffix tensors
        self.ttne_label_idx = self.suffix_num_tensor_idx + 1
        self.rrt_label_idx = self.ttne_label_idx + 1
        self.act_label_idx = self.rrt_label_idx + 1
        # Add outcome index if applicable

        # --- Embedding Layers ---
        self.embeddings = nn.ModuleDict()
        total_feature_dim_enc = 0
        total_feature_dim_dec = 0

        # Activity Embedding (Shared) - uses activity vocab size which includes EOS/END
        # The input activity IDs should NOT include EOS/END, only mapped activities + pad + unk
        act_map_size = max(metadata['categorical_mapping_dict'][self.activity_col].values()) + 1 # Size based on mapping
        # Vocab size for embedding layer needs to account for padding (0)
        act_vocab_size = act_map_size + 1 # +1 for potential OOV if its ID is max+1
        self.embeddings[self.activity_col] = nn.Embedding(act_vocab_size, d_model // 2, padding_idx=self.pad_id) # Allocate roughly half to activity
        total_feature_dim_enc += self.embeddings[self.activity_col].embedding_dim
        total_feature_dim_dec += self.embeddings[self.activity_col].embedding_dim
        print(f"Shared Embedding '{self.activity_col}': Dim={self.embeddings[self.activity_col].embedding_dim}, Vocab={act_vocab_size}")

        # Other Categorical Embeddings (Prefix Only in this example)
        num_other_cat = self.num_prefix_cat_tensors - 1 # Exclude activity
        if num_other_cat > 0:
            remaining_dim_enc = d_model - total_feature_dim_enc - (d_model // 4 if len(self.prefix_num_cols) > 0 else 0) # Reserve some dim for nums
            base_embed_dim_cat = max(1, remaining_dim_enc // num_other_cat)
            for col in self.prefix_cat_cols:
                if col == self.activity_col: continue # Skip shared activity embedding
                map_size = max(metadata['categorical_mapping_dict'][col].values()) + 1
                vocab_size = map_size + 1 # +1 for potential OOV
                embed_dim = base_embed_dim_cat
                # Prefix features have their own key like 'feat_{col}'
                feat_key = f'feat_{col}'
                self.embeddings[feat_key] = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_id)
                total_feature_dim_enc += embed_dim
                print(f"Prefix Embedding '{feat_key}': Dim={embed_dim}, Vocab={vocab_size}")

        # Numerical Feature Projection (Prefix)
        num_num_feats_pref = len(self.prefix_num_cols)
        if num_num_feats_pref > 0:
             num_proj_dim_enc = d_model - total_feature_dim_enc
             if num_proj_dim_enc <= 0: num_proj_dim_enc = d_model // 8; print("Warn: Prefix Embeddings fill d_model")
             self.numerical_projector_enc = nn.Linear(num_num_feats_pref, num_proj_dim_enc)
             total_feature_dim_enc += num_proj_dim_enc
             print(f"Prefix Numerical Projection: Dim={num_proj_dim_enc} (from {num_num_feats_pref} features)")
        else: self.numerical_projector_enc = None

        # Numerical Feature Projection (Suffix) - only ts_prev, ts_start
        num_num_feats_suff = len(self.suffix_num_cols)
        if num_num_feats_suff > 0:
             num_proj_dim_dec = d_model - total_feature_dim_dec
             if num_proj_dim_dec <= 0: num_proj_dim_dec = d_model // 8; print("Warn: Suffix Embeddings fill d_model")
             self.numerical_projector_dec = nn.Linear(num_num_feats_suff, num_proj_dim_dec)
             total_feature_dim_dec += num_proj_dim_dec
             print(f"Suffix Numerical Projection: Dim={num_proj_dim_dec} (from {num_num_feats_suff} features)")
        else: self.numerical_projector_dec = None

        # --- Input Projections ---
        self.input_proj_enc = nn.Linear(total_feature_dim_enc, d_model) if total_feature_dim_enc != d_model else nn.Identity()
        self.input_proj_dec = nn.Linear(total_feature_dim_dec, d_model) if total_feature_dim_dec != d_model else nn.Identity()
        print(f"Final Encoder Input Projection: {total_feature_dim_enc} -> {d_model}")
        print(f"Final Decoder Input Projection: {total_feature_dim_dec} -> {d_model}")

        # --- Positional Encoding ---
        self.positional_encoding = LearnedPositionalEmbedding(self.padding_length + 1, d_model, self.pad_id)

        # --- Encoder & Decoder ---
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_enc_layers, norm=nn.LayerNorm(d_model))
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_dec_layers, norm=nn.LayerNorm(d_model))

        # --- Output Heads ---
        # Activity head output size should match the labels (incl. EOS + Pad)
        final_act_vocab_size = metadata['activity_vocab_size_incl_pad']
        self.output_proj = nn.Linear(d_model, final_act_vocab_size)
        # Time head output size matches number of time labels (tt_next, rtime)
        num_time_labels = len(self.time_label_features)
        self.time_head = nn.Linear(d_model, num_time_labels) if num_time_labels > 0 else None
        print(f"Output Heads: Activity (size {final_act_vocab_size}), Time (size {num_time_labels})")

        self.dropout = nn.Dropout(dropout)
# 在 train_op.py 的 AdaptedBartModel 类中

    def _prepare_encoder_input(self, inputs_tuple):
        """Combines prefix features from the input tuple."""
        combined_enc = []
        # 检查 inputs_tuple 是否为空或长度不足
        if not inputs_tuple or not isinstance(inputs_tuple[0], torch.Tensor):
             print("Error: Invalid inputs_tuple in _prepare_encoder_input")
             # 返回空的或者默认的 Tensor，或者抛出异常
             # 这里返回空的，可能需要在调用处处理
             return torch.empty(0), torch.empty(0)

        batch_size = inputs_tuple[0].size(0)
        seq_len = inputs_tuple[0].size(1)
        device = inputs_tuple[0].device

        # Activity Embedding (Index known based on prefix_cat_cols)
        # 确保 activity_col 在 prefix_cat_cols 中
        try:
            activity_tensor_idx = self.prefix_cat_cols.index(self.activity_col)
            if activity_tensor_idx >= len(inputs_tuple):
                 raise IndexError(f"Activity tensor index {activity_tensor_idx} out of bounds for inputs_tuple length {len(inputs_tuple)}")
            act_ids = inputs_tuple[activity_tensor_idx]
            combined_enc.append(self.embeddings[self.activity_col](act_ids))
        except ValueError:
            print(f"Error: Activity column '{self.activity_col}' not found in prefix_cat_cols.")
            # 可能需要返回错误或默认值
            return torch.empty(0), torch.empty(0)
        except IndexError as e:
            print(f"Error accessing activity tensor: {e}")
            return torch.empty(0), torch.empty(0)
        except KeyError:
            print(f"Error: Embedding for '{self.activity_col}' not found.")
            return torch.empty(0), torch.empty(0)


        # Other Categorical Embeddings (Prefix)
        cat_idx_offset = 0
        for i, col in enumerate(self.prefix_cat_cols):
            if col == self.activity_col:
                cat_idx_offset = 1 # Account for activity already processed
                continue
            feat_key = f'feat_{col}'
            # 预期索引应该基于非活动分类特征的顺序
            # 假设分类张量紧跟活动张量之后
            tensor_idx = self.prefix_cat_cols.index(col) # 获取该特征在原始列表中的索引

            if tensor_idx < len(inputs_tuple) and feat_key in self.embeddings:
                 try:
                     combined_enc.append(self.embeddings[feat_key](inputs_tuple[tensor_idx]))
                 except IndexError as e:
                     print(f"Error accessing tensor for {feat_key} at index {tensor_idx}: {e}")
                     # 可以选择添加零向量或返回错误
                     emb_layer = self.embeddings[feat_key]
                     combined_enc.append(torch.zeros(batch_size, seq_len, emb_layer.embedding_dim, device=device, dtype=emb_layer.weight.dtype))
                 except KeyError:
                     print(f"Error: Embedding for {feat_key} not found (should not happen if check passed).")
                     # 添加零向量
                     combined_enc.append(torch.zeros(batch_size, seq_len, 10, device=device)) # 假设一个默认维度

            elif feat_key in self.embeddings: # 如果索引越界但嵌入层存在
                 print(f"Warning: Tensor index {tensor_idx} out of bounds for feature '{col}', but embedding exists. Adding zeros.")
                 emb_layer = self.embeddings[feat_key]
                 combined_enc.append(torch.zeros(batch_size, seq_len, emb_layer.embedding_dim, device=device, dtype=emb_layer.weight.dtype))
            # else: # 如果嵌入层不存在，则不添加任何东西
            #    print(f"Warning: Embedding key {feat_key} not found during enc prep.")


        # Numerical Features (Prefix)
        if self.numerical_projector_enc:
            # 检查索引是否有效
            if self.prefix_num_tensor_idx >= len(inputs_tuple):
                 print(f"Error: Prefix numerical tensor index {self.prefix_num_tensor_idx} out of bounds for inputs_tuple length {len(inputs_tuple)}")
                 # 添加零向量或返回错误
                 combined_enc.append(torch.zeros(batch_size, seq_len, self.numerical_projector_enc.out_features, device=device))
            else:
                num_tensor = inputs_tuple[self.prefix_num_tensor_idx] # Get tensor by index
                if not isinstance(num_tensor, torch.Tensor):
                    print(f"Error: Expected a Tensor at prefix numerical index {self.prefix_num_tensor_idx}, got {type(num_tensor)}")
                    combined_enc.append(torch.zeros(batch_size, seq_len, self.numerical_projector_enc.out_features, device=device))
                else:
                    # --- 在这里添加裁剪 ---
                    # 限制标准化后的数值特征范围，例如 [-20, 20]
                    num_tensor_clipped = torch.clamp(num_tensor, min=-20.0, max=20.0)
                    # 检查裁剪后的张量中是否还包含 NaN 或 Inf
                    if not torch.isfinite(num_tensor_clipped).all():
                         print(f"Warning: Non-finite values remain in prefix numerical tensor AFTER clipping in batch.")
                         # 可以选择用0填充 NaN/Inf
                         num_tensor_clipped = torch.nan_to_num(num_tensor_clipped, nan=0.0, posinf=20.0, neginf=-20.0)
                    combined_enc.append(self.numerical_projector_enc(num_tensor_clipped))

        elif len(self.prefix_num_cols) > 0: # Check if nums expected but no projector
             print(f"Warning: Numerical prefix features defined but no encoder projector.")


        # Concatenate and Project
        try:
            if not combined_enc: # 如果列表为空
                 print("Error: combined_enc list is empty in _prepare_encoder_input.")
                 return torch.empty(0), torch.empty(0)
            # 检查拼接前各张量维度是否一致（除了最后一个维度）
            ref_shape = combined_enc[0].shape[:-1]
            if not all(t.shape[:-1] == ref_shape for t in combined_enc):
                 print(f"Error: Mismatched batch/sequence dimensions before encoder concatenation:")
                 for i, t in enumerate(combined_enc): print(f" Tensor {i}: {t.shape}")
                 return torch.empty(0), torch.empty(0)

            concatenated_enc = torch.cat(combined_enc, dim=-1)
            projected_enc = self.input_proj_enc(concatenated_enc)
        except Exception as e:
            print(f"Error during encoder concatenation/projection: {e}")
            print("Shapes before cat:", [t.shape for t in combined_enc])
            return torch.empty(0), torch.empty(0)

        return projected_enc, act_ids # Return projected and activity IDs for pos encoding

    def _prepare_decoder_input(self, inputs_tuple, shift_right=True):
        """Combines suffix features from the input tuple."""
        combined_dec = []

        # Determine tensor indices based on structure
        act_tensor_idx = self.suffix_cat_tensor_start_idx
        num_tensor_idx = self.suffix_num_tensor_idx

        # 检查索引是否有效
        if act_tensor_idx >= len(inputs_tuple) or num_tensor_idx >= len(inputs_tuple):
             print(f"Error: Suffix tensor indices out of bounds (Act: {act_tensor_idx}, Num: {num_tensor_idx}, Total: {len(inputs_tuple)})")
             return torch.empty(0), torch.empty(0)

        act_ids = inputs_tuple[act_tensor_idx]
        num_features = inputs_tuple[num_tensor_idx]

        # 检查获取的是否是 Tensor
        if not isinstance(act_ids, torch.Tensor) or not isinstance(num_features, torch.Tensor):
            print(f"Error: Expected Tensors for suffix features, got {type(act_ids)} and {type(num_features)}")
            return torch.empty(0), torch.empty(0)


        if shift_right:
             # Teacher forcing: shift inputs right, prepend SOS (use PAD as SOS placeholder)
             # 检查 num_features 是否至少有3个维度
             if num_features.ndim < 3:
                  print(f"Error: Suffix numerical features tensor has unexpected shape {num_features.shape}. Expected 3 dimensions.")
                  return torch.empty(0), torch.empty(0)
             sos_shape = (act_ids.size(0), 1)
             device = act_ids.device
             sos_acts = torch.full(sos_shape, self.pad_id, dtype=torch.long, device=device) # Use PAD as SOS
             sos_nums = torch.zeros(act_ids.size(0), 1, num_features.size(2), dtype=torch.float, device=device)

             act_ids_shifted = torch.cat([sos_acts, act_ids[:, :-1]], dim=1)
             num_features_shifted = torch.cat([sos_nums, num_features[:, :-1, :]], dim=1)
        else:
             # Used during AR generation (input is already prepared)
             act_ids_shifted = act_ids
             num_features_shifted = num_features

        batch_size, seq_len = act_ids_shifted.shape

        # Activity Embedding
        try:
            combined_dec.append(self.embeddings[self.activity_col](act_ids_shifted))
        except KeyError:
            print(f"Error: Embedding for '{self.activity_col}' not found.")
            return torch.empty(0), torch.empty(0)

        # Numerical Features (Suffix)
        if self.numerical_projector_dec:
             # --- 在这里添加裁剪 ---
             # 对移位后的 decoder 数值输入进行裁剪
             num_features_shifted_clipped = torch.clamp(num_features_shifted, min=-20.0, max=20.0)
             # 检查裁剪后的张量中是否还包含 NaN 或 Inf
             if not torch.isfinite(num_features_shifted_clipped).all():
                  print(f"Warning: Non-finite values remain in suffix numerical tensor AFTER clipping in batch.")
                  num_features_shifted_clipped = torch.nan_to_num(num_features_shifted_clipped, nan=0.0, posinf=20.0, neginf=-20.0)
             # 使用裁剪后的张量进行投影
             combined_dec.append(self.numerical_projector_dec(num_features_shifted_clipped))
        elif len(self.suffix_num_cols) > 0:
             print(f"Warning: Numerical suffix features defined but no decoder projector.")

        # Concatenate and Project
        try:
            if not combined_dec:
                 print("Error: combined_dec list is empty in _prepare_decoder_input.")
                 return torch.empty(0), torch.empty(0)
             # 检查拼接前各张量维度
            ref_shape = combined_dec[0].shape[:-1]
            if not all(t.shape[:-1] == ref_shape for t in combined_dec):
                  print(f"Error: Mismatched batch/sequence dimensions before decoder concatenation:")
                  for i, t in enumerate(combined_dec): print(f" Tensor {i}: {t.shape}")
                  return torch.empty(0), torch.empty(0)

            concatenated_dec = torch.cat(combined_dec, dim=-1)
            projected_dec = self.input_proj_dec(concatenated_dec)
        except Exception as e:
            print(f"Error during decoder concatenation/projection: {e}")
            print("Shapes before cat:", [t.shape for t in combined_dec])
            return torch.empty(0), torch.empty(0)

        return projected_dec, act_ids_shifted # Return projected and ACT IDs for pos encoding
    @staticmethod
    def _generate_square_subsequent_mask(sz, device):
        return torch.triu(torch.full((sz, sz), True, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, inputs_tuple):
        """ Processes input tuple. """
        # 1. Prepare Encoder Input
        enc_combined_features, enc_act_ids = self._prepare_encoder_input(inputs_tuple)
        enc_input = self.dropout(enc_combined_features + self.positional_encoding(enc_act_ids.shape))
        enc_padding_mask = inputs_tuple[self.prefix_padding_mask_idx] # Get mask by index

        # 2. Encode
        memory = self.encoder(enc_input, src_key_padding_mask=enc_padding_mask)

        # 3. Prepare Decoder Input (Shifted Right for Teacher Forcing)
        dec_combined_features, dec_act_ids_shifted = self._prepare_decoder_input(inputs_tuple, shift_right=True)
        dec_input = self.dropout(dec_combined_features + self.positional_encoding(dec_act_ids_shifted.shape))
        dec_padding_mask = (dec_act_ids_shifted == self.pad_id) # Padding mask for shifted decoder input
        L_tgt = dec_input.size(1)
        device = dec_input.device
        dec_attn_mask = self._generate_square_subsequent_mask(L_tgt, device) # Causal mask

        # 4. Decode
        dec_output = self.decoder(
            tgt=dec_input, memory=memory,
            tgt_mask=dec_attn_mask,
            memory_mask=None,
            tgt_key_padding_mask=dec_padding_mask,
            memory_key_padding_mask=enc_padding_mask
        )

        # 5. Output Projections
        activity_logits = self.output_proj(dec_output) # (B, L_target-1, V_act_final)
        time_preds = self.time_head(dec_output) if self.time_head else None # (B, L_target-1, N_time)

        # --- NOTE: RRT/Outcome Heads (if applicable) ---
        # These typically use the encoder output ('memory') or a specific decoder state.
        # Author's SuTraN uses decoder output. Let's assume that for now.
        # But they are only calculated based on the FIRST output step.
        # Here we return the full sequence, loss function needs to handle slicing.
        # A cleaner way might be to calculate them separately using encoder memory.
        rrt_preds = None # Placeholder
        if 'rtime' in self.time_label_features and self.time_head:
             # Placeholder: For simplicity, return the time head output.
             # A dedicated head might be better. Loss needs to select correct index.
             rrt_preds = time_preds # Loss needs to select index corresponding to 'rtime'

        # TODO: Add outcome head if self.outcome_bool is True

        return activity_logits, time_preds # TODO: Return rrt_preds, out_preds if implemented

# --- Multi-Output Loss Function (Adapted for Tuples) ---
class SutranLikeLossTuple(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.pad_id = metadata['pad_id']
        self.time_label_features = metadata.get('time_label_features', [])
        # Activity vocab size needs to match the output projection layer
        self.act_vocab_size = metadata['activity_vocab_size_incl_pad']
        self.criterion_act = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.criterion_time = nn.L1Loss(reduction='none') # MAE loss per element

        # --- Determine Label Indices in Label Tuple ---
        # This MUST match the order returned by generate_label_tensors
        self.label_idx_map = {}
        current_idx = 0
        if 'tt_next' in self.time_label_features:
             self.label_idx_map['tt_next'] = current_idx; current_idx += 1
        if 'rtime' in self.time_label_features:
             self.label_idx_map['rtime'] = current_idx; current_idx += 1
        self.label_idx_map['activity'] = current_idx; current_idx += 1
        # Add outcome index if applicable

        # --- Determine Prediction Indices in Output Tuple ---
        # This MUST match the return order of AdaptedBartModel.forward
        self.pred_idx_map = {'activity': 0, 'time': 1} # Base
        # TODO: Add RRT/Outcome indices if model returns them


    def forward(self, model_outputs_tuple, labels_tuple):
        # Unpack predictions (assuming order: act_logits, time_preds, [opt: rrt_pred], [opt: out_pred])
        activity_logits = model_outputs_tuple[self.pred_idx_map['activity']]
        time_preds = model_outputs_tuple[self.pred_idx_map['time']] # This contains ALL time predictions (ttne, rtime)
        # TODO: Get rrt_pred, out_pred if applicable

        # Unpack labels (assuming order: ttne_labels, rrt_labels, act_labels, [opt: out_labels])
        target_act = labels_tuple[self.label_idx_map['activity']]
        # Combine time labels for comparison
        target_time_list = []
        if 'tt_next' in self.label_idx_map: target_time_list.append(labels_tuple[self.label_idx_map['tt_next']])
        if 'rtime' in self.label_idx_map: target_time_list.append(labels_tuple[self.label_idx_map['rtime']])
        target_time = torch.cat(target_time_list, dim=-1) if target_time_list else None # (B, L_target, N_time)

        device = activity_logits.device
        target_act = target_act.to(device)
        if target_time is not None: target_time = target_time.to(device)


        # Align lengths (model outputs are L_target - 1)
        L_pred = activity_logits.size(1)
        target_act = target_act[:, :L_pred] # Compare with first L_pred labels
        if target_time is not None: target_time = target_time[:, :L_pred, :]


        # --- Activity Loss ---
        # Reshape for CrossEntropyLoss: (B * L, V) and (B * L)
        act_loss = self.criterion_act(
            activity_logits.reshape(-1, self.act_vocab_size),
            target_act.reshape(-1)
        )

        # --- Time Loss (Combined TTNE and RRT if applicable) ---
        if time_preds is not None and target_time is not None and len(self.time_label_features) > 0:
            # Ensure time_preds has the same number of features as target_time
            if time_preds.size(-1) != target_time.size(-1):
                 print(f"Warning: Mismatch time_preds ({time_preds.size(-1)}) vs target_time ({target_time.size(-1)}) features.")
                 # Attempt to slice time_preds if it has more features than expected
                 if time_preds.size(-1) > target_time.size(-1):
                      time_preds = time_preds[..., :target_time.size(-1)]
                 else: # Cannot compute loss if target has more features
                      avg_time_loss = torch.tensor(0.0).to(device)
                      print("Skipping time loss calculation due to feature mismatch.")
                      return act_loss, act_loss.item(), 0.0 # Return only act_loss

            # Create mask based on target activity padding
            time_loss_mask = (target_act != self.pad_id).unsqueeze(-1).expand_as(time_preds) # (B, L_pred, N_time)

            # Create mask for time label padding (-100)
            time_pad_mask = (target_time != -100.0) # (B, L_pred, N_time)

            # Combine masks: loss calculated only where target activity is not pad AND target time is not -100
            final_time_mask = time_loss_mask & time_pad_mask

            elementwise_time_loss = self.criterion_time(time_preds, target_time) # (B, L_pred, N_time)
            masked_time_loss = elementwise_time_loss * final_time_mask # Apply combined mask
            # Average over non-masked elements
            num_valid_time_elements = final_time_mask.sum().clamp(min=1e-9)
            avg_time_loss = masked_time_loss.sum() / num_valid_time_elements
        else:
            avg_time_loss = torch.tensor(0.0).to(device)

        # --- Combine Losses ---
        # Simple sum for now, consider weighting (e.g., lambda_time=0.5)
        # combined_loss = act_loss + lambda_time * avg_time_loss
        combined_loss = act_loss + avg_time_loss

        # TODO: Add RRT loss (maybe MAE on first step?) and Outcome loss (BCE) if applicable

        # Return total loss and individual components for logging
        # Need to separate ttne/rrt loss items if calculated together in avg_time_loss
        # For now, just return combined time loss item
        return combined_loss, act_loss.item(), avg_time_loss.item()


# 在 train_op.py 文件中

# --- Evaluation Metrics and Function (Adapted for Tuples) ---
# Keep DLS function
def damerau_levenshtein_similarity(p, t):
    """ Calculates Damerau-Levenshtein similarity between two sequences. """
    # Ensure inputs are tuples of strings for the library
    if not isinstance(p, (list, tuple, str)): p = tuple(map(str, p))
    elif not isinstance(p, tuple): p = tuple(p)
    if not isinstance(t, (list, tuple, str)): t = tuple(map(str, t))
    elif not isinstance(t, tuple): t = tuple(t)

    # Handle empty sequences gracefully
    if not p and not t: return 1.0
    if not p or not t: return 0.0

    d = DamerauLevenshtein.distance(p, t); m = max(len(p), len(t))
    # Add check for m > 0 before division
    return 1.0 - (d / m) if m > 0 else 1.0

# Keep MAE function
def mean_absolute_error(p,t,inv=False):
    """ Calculates Mean Absolute Error, assumes inputs are valid (non-padded, de-standardized). """
    # Ensure inputs are numpy arrays
    pn,tn = np.array(p,dtype=np.float64),np.array(t,dtype=np.float64)

    # Basic check for NaN/inf in inputs which might occur after de-standardization/inverse-log
    valid_mask = np.isfinite(pn) & np.isfinite(tn)
    pf,tf = pn[valid_mask],tn[valid_mask]

    if len(pf)==0: return 0.0 # Return 0 if no valid points to compare

    # Inverse transform (log) should be handled before calling this function
    pi,ti = pf, tf

    return np.mean(np.abs(pi-ti))


# --- Modified Evaluation Function ---
def evaluate_model_tuple(model, dataloader, device, loss_fn, metadata):
    """Evaluates model using teacher forcing with TUPLE data, calculates DLS and MAE."""
    model.eval() # Set model to evaluation mode
    total_loss, total_act_loss, total_time_loss = 0.0, 0.0, 0.0 # Use floats for accumulation
    all_pred_event_seqs, all_target_event_seqs = [], []
    all_pred_time_lists = {label: [] for label in metadata['time_label_features']}
    all_target_time_lists = {label: [] for label in metadata['time_label_features']}
    num_samples = 0
    num_valid_batches = 0 # Count batches where loss calculation was successful

    # Load necessary metadata
    pad_id = metadata['pad_id']
    eos_token_int = metadata.get('eos_token_int', -1) # Get EOS integer ID, default to -1 if missing
    if eos_token_int == -1: print("Warning: 'eos_token_int' not found in metadata.")
    time_label_features = metadata.get('time_label_features', [])
    label_idx_map = loss_fn.label_idx_map # Use map from loss function
    pred_idx_map = loss_fn.pred_idx_map
    num_label_tensors = len(label_idx_map)
    log_transformed_cols = metadata.get('log_transformed_cols',[]) # Get log transformed cols info

    # Create reverse map for activity IDs to names (handle potential errors)
    try: id_to_event_map = {v: k for k, v in metadata['categorical_mapping_dict'][metadata['activity_col']].items()}
    except Exception as e: id_to_event_map = {}; print(f"Warn: Failed to create id_to_event map: {e}")
    # Ensure END_TOKEN is defined globally or passed correctly
    global END_TOKEN # Or pass it as an argument if preferred
    if eos_token_int != -1 and 'END_TOKEN' in globals():
        id_to_event_map[eos_token_int] = END_TOKEN # Add mapping for EOS using the defined constant
    elif eos_token_int != -1:
        print("Warning: END_TOKEN constant not found, cannot map EOS integer ID.")


    print("Running evaluation (Teacher Forcing)...")
    with torch.no_grad(): # Disable gradient calculations
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for i, batch_tuple in enumerate(pbar):
            batch_valid = True # Flag to track if batch processing is successful
            # --- Move batch to device ---
            try:
                batch_device_tuple = tuple(t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in batch_tuple)
            except Exception as e:
                print(f"Error moving batch {i} to device: {e}. Skipping batch.")
                batch_valid = False
                continue # Skip this batch

            current_batch_size = batch_device_tuple[0].size(0) # Size from first tensor
            num_samples += current_batch_size

            # Separate inputs and labels
            inputs_device = batch_device_tuple[:-num_label_tensors]
            labels_device = batch_device_tuple[-num_label_tensors:]

            # <<< --- Optional Debug Prints for Inputs (Uncomment if needed) --- >>>
            # if i < 2: # Print for first few batches
            #     print(f"\n--- Debugging Batch {i} ---")
            #     print("Input shapes:", [t.shape for t in inputs_device])
            #     print("Input NaNs:", [torch.isnan(t).any().item() if isinstance(t, torch.Tensor) else 'Not Tensor' for t in inputs_device])
            #     print("Input Infs:", [torch.isinf(t).any().item() if isinstance(t, torch.Tensor) else 'Not Tensor' for t in inputs_device])
            #     print("Label shapes:", [t.shape for t in labels_device])
            #     print("Label NaNs:", [torch.isnan(t).any().item() if isinstance(t, torch.Tensor) else 'Not Tensor' for t in labels_device])
            #     print("Label Infs:", [torch.isinf(t).any().item() if isinstance(t, torch.Tensor) else 'Not Tensor' for t in labels_device])

            # --- Get Model Predictions ---
            try:
                model_outputs = model(inputs_device)
                activity_logits = model_outputs[pred_idx_map['activity']]
                time_preds = model_outputs[pred_idx_map.get('time', -1)] # Use .get for safety

                # <<< --- Optional Debug Prints for Model Outputs (Uncomment if needed) --- >>>
                # if i < 2: # Print for first few batches
                #     print("Model Output Shapes:", [t.shape if t is not None else 'None' for t in model_outputs])
                #     print("Model Output NaNs:", [torch.isnan(t).any().item() if t is not None else 'None' for t in model_outputs])
                #     print("Model Output Infs:", [torch.isinf(t).any().item() if t is not None else 'None' for t in model_outputs])
                #     if activity_logits is not None:
                #          print("Activity Logits Max/Min:", activity_logits.max().item(), activity_logits.min().item())
                #     if time_preds is not None:
                #          print("Time Preds Max/Min:", time_preds.max().item(), time_preds.min().item())

                # Check for NaNs/Infs in model output
                if not torch.isfinite(activity_logits).all() or (time_preds is not None and not torch.isfinite(time_preds).all()):
                   print(f"Warning: Non-finite values detected in model output for batch {i}. Skipping loss & metrics for this batch.")
                   batch_valid = False # Mark batch as invalid for loss accumulation and metrics

            except Exception as e:
                print(f"Error during model forward pass for batch {i}: {e}")
                batch_valid = False
                continue # Skip this batch

            # --- Calculate Loss (only if batch is valid so far) ---
            if batch_valid:
                try:
                    # <<< --- Optional Debug Prints for Loss Inputs (Uncomment if needed) --- >>>
                    # if i < 2: # Print for first few batches
                    #    print("Inputs to Loss Fn - Model Output NaNs:", [torch.isnan(t).any().item() if t is not None else 'None' for t in model_outputs])
                    #    print("Inputs to Loss Fn - Labels NaNs:", [torch.isnan(t).any().item() if isinstance(t, torch.Tensor) else 'Not Tensor' for t in labels_device])

                    loss, act_loss, time_loss = loss_fn(model_outputs, labels_device)

                    if not torch.isfinite(loss):
                        print(f"Warning: Non-finite loss ({loss.item()}) calculated for batch {i}. Act: {act_loss}, Time: {time_loss}. Skipping loss accumulation.")
                        batch_valid = False # Mark as invalid if loss is non-finite
                    else:
                        total_loss += loss.item() * current_batch_size
                        total_act_loss += act_loss * current_batch_size
                        total_time_loss += time_loss * current_batch_size
                        num_valid_batches += 1 # Increment count of batches with valid loss

                except Exception as e:
                    print(f"Error during loss calculation for batch {i}: {e}")
                    batch_valid = False # Mark as invalid if loss calculation fails

            # --- Prepare for Metrics (only if batch is still valid) ---
            if batch_valid:
                try:
                    # Move necessary tensors to CPU for metric calculation
                    target_act_labels = labels_device[label_idx_map['activity']].cpu() # (B, L_target)

                    # Combine target time labels on CPU, ensuring they are 3D
                    target_time_list_cpu = []
                    if 'tt_next' in label_idx_map:
                        tt_next_tensor = labels_device[label_idx_map['tt_next']].cpu()
                        if tt_next_tensor.ndim == 2: tt_next_tensor = tt_next_tensor.unsqueeze(-1)
                        elif tt_next_tensor.ndim != 3: print(f"Warning: Unexpected dimension {tt_next_tensor.ndim} for tt_next."); batch_valid=False; continue
                        target_time_list_cpu.append(tt_next_tensor)
                    if 'rtime' in label_idx_map:
                        rtime_tensor = labels_device[label_idx_map['rtime']].cpu()
                        if rtime_tensor.ndim == 2: rtime_tensor = rtime_tensor.unsqueeze(-1)
                        elif rtime_tensor.ndim != 3: print(f"Warning: Unexpected dimension {rtime_tensor.ndim} for rtime."); batch_valid=False; continue
                        target_time_list_cpu.append(rtime_tensor)

                    target_times_standardized = torch.cat(target_time_list_cpu, dim=-1) if target_time_list_cpu else None # (B, L_target, N_time)

                    pred_act_ids = torch.argmax(activity_logits, dim=-1).cpu() # (B, L_pred)
                    pred_times_standardized = time_preds.cpu() if time_preds is not None else None # (B, L_pred, N_time)

                    # Align lengths (predictions are L_target - 1)
                    L_pred = pred_act_ids.size(1)
                    target_act_ids = target_act_labels[:, :L_pred]
                    if target_times_standardized is not None: target_times_standardized_aligned = target_times_standardized[:, :L_pred, :]
                    else: target_times_standardized_aligned = None

                    target_mask = (target_act_ids != pad_id) # Mask for valid target steps (B, L_pred)

                    # --- Collect sequences and times for metric calculation ---
                    for sample_idx in range(target_act_ids.size(0)):
                        actual_len = target_mask[sample_idx].sum().item() # Length of valid target sequence
                        if actual_len == 0: continue # Skip if target sequence is empty/all padding

                        # Get predicted and target activity IDs up to actual length
                        pi = pred_act_ids[sample_idx][:actual_len].tolist()
                        ti = target_act_ids[sample_idx][:actual_len].tolist()

                        # Convert IDs to event names (handle potential unknowns/EOS)
                        pe = [id_to_event_map.get(x, f'<UNK_{x}>') for x in pi]
                        te = [id_to_event_map.get(x, f'<UNK_{x}>') for x in ti]
                        all_pred_event_seqs.append(pe)
                        all_target_event_seqs.append(te)

                        # Collect time predictions and targets up to actual length
                        if pred_times_standardized is not None and target_times_standardized_aligned is not None:
                            # Get standardized predictions and targets for this sample
                            pred_t_std = pred_times_standardized[sample_idx][:actual_len, :].numpy()
                            targ_t_std = target_times_standardized_aligned[sample_idx][:actual_len, :].numpy()

                            for time_idx, label_name in enumerate(time_label_features):
                                # Get standardized prediction and target values for this specific time feature
                                pred_vals_std = pred_t_std[:, time_idx]
                                targ_vals_std = targ_t_std[:, time_idx]

                                # Create mask based on the standardized target value (-100.0)
                                valid_target_mask = targ_vals_std != -100.0

                                # Proceed only if there are valid targets to compare
                                if valid_target_mask.any():
                                    # De-standardize ONLY the valid values for comparison
                                    mean = metadata['train_means'].get(label_name, 0)
                                    std = metadata['train_stds'].get(label_name, 1)
                                    if std < 1e-9: std = 1.0 # Avoid division by zero if std is near zero

                                    # Get the valid values using the mask
                                    pred_vals_valid_std = pred_vals_std[valid_target_mask]
                                    targ_vals_valid_std = targ_vals_std[valid_target_mask]

                                    # De-standardize
                                    pred_vals_valid = pred_vals_valid_std * std + mean
                                    targ_vals_valid = targ_vals_valid_std * std + mean

                                    # Inverse log transform if applied during preprocessing
                                    if label_name in log_transformed_cols:
                                        # Ensure values are reasonable before expm1
                                        pred_vals_valid = np.expm1(np.minimum(pred_vals_valid, 30)) # Cap input to avoid overflow
                                        targ_vals_valid = np.expm1(np.minimum(targ_vals_valid, 30))

                                    # Append the valid, de-standardized values
                                    all_pred_time_lists[label_name].extend(pred_vals_valid)
                                    all_target_time_lists[label_name].extend(targ_vals_valid)
                except Exception as e:
                    print(f"Error during metric preparation/collection for batch {i}, sample {sample_idx if 'sample_idx' in locals() else 'N/A'}: {e}")
                    # Continue to next batch or handle as needed


    # --- Calculate Average Metrics ---
    # Use num_valid_batches for averaging loss to avoid dilution by failed batches
    if num_valid_batches == 0: num_valid_batches = 1e-9 # Avoid division by zero

    avg_loss = total_loss / (num_valid_batches * dataloader.batch_size) if isinstance(dataloader.batch_size, int) else total_loss / num_samples # Approx avg loss per sample across valid batches
    avg_act_loss = total_act_loss / (num_valid_batches * dataloader.batch_size) if isinstance(dataloader.batch_size, int) else total_act_loss / num_samples
    avg_time_loss = total_time_loss / (num_valid_batches * dataloader.batch_size) if isinstance(dataloader.batch_size, int) else total_time_loss / num_samples

    # Calculate DLS, handle empty lists
    mean_dls = np.mean([damerau_levenshtein_similarity(p, t) for p, t in zip(all_pred_event_seqs, all_target_event_seqs)]) if all_pred_event_seqs else 0.0

    # Calculate MAE, handle empty lists
    mae_results = {}
    for label_name in time_label_features:
        pred_list = all_pred_time_lists.get(label_name, [])
        targ_list = all_target_time_lists.get(label_name, [])
        if not pred_list or not targ_list:
             mae = float('nan') # Report NaN if no valid data points for MAE
             print(f"Warning: Empty prediction or target list for MAE calculation of {label_name}")
        else:
             # MAE calculation should receive de-standardized, non-padded values
             mae_seconds = mean_absolute_error(pred_list, targ_list, inv=False)
             mae = mae_seconds / 60.0 if np.isfinite(mae_seconds) else float('nan') # Convert seconds MAE to minutes, handle potential NaN from MAE func
        mae_results[label_name] = mae

    print(f"\nEvaluation Results (Teacher Forcing):")
    print(f"  Avg Loss (valid batches): {avg_loss:.4f} (Act: {avg_act_loss:.4f}, Time: {avg_time_loss:.4f})")
    print(f"  Activity DLS: {mean_dls:.4f}")
    for label, mae in mae_results.items():
        print(f"  {label} MAE (min): {mae:.4f}")

    model.train() # Set back to train mode
    return mean_dls, mae_results # Return key metrics

def run_training_tuple(model, train_loader, val_loader, optimizer, lr_scheduler, loss_fn, device, epochs, metadata,
                       checkpoint_dir, best_model_path, start_epoch=0, initial_best_dls=-1.0, save_every=10, evaluate_every=1, grad_clip=1.0, use_amp=False):
    """ Main training loop modified for tuple data format. """
    print(f"--- Starting Training (Epochs {start_epoch+1}-{epochs}) ---")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_dls = initial_best_dls
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Determine number of label tensors expected by loss function
    num_label_tensors = len(loss_fn.label_idx_map)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_c_loss, total_t_loss = 0, 0, 0
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train", leave=False)

        for batch_tuple in pbar:
            # Move batch to device
            batch_device_tuple = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch_tuple)

            # Separate inputs and labels based on expected number of label tensors
            inputs_device = batch_device_tuple[:-num_label_tensors]
            labels_device = batch_device_tuple[-num_label_tensors:]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Pass only input part of the tuple to the model
                model_outputs = model(inputs_device)
                # Loss function expects model outputs and label part of the tuple
                loss, act_loss, time_loss = loss_fn(model_outputs, labels_device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_c_loss += act_loss
            total_t_loss += time_loss
            pbar.set_postfix({'L':f'{loss.item():.3f}','ActL':f'{act_loss:.3f}','TimeL':f'{time_loss:.3f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_c = total_c_loss / num_batches if num_batches > 0 else 0
        avg_t = total_t_loss / num_batches if num_batches > 0 else 0
        print(f"\nEp {epoch+1} Avg Loss: {avg_loss:.4f} (Act:{avg_c:.4f}, Time:{avg_t:.4f})")

        # --- Validation ---
        if (epoch + 1) % evaluate_every == 0:
            # Use the tuple-adapted evaluation function
            current_dls, current_mae_dict = evaluate_model_tuple(model, val_loader, device, loss_fn, metadata)
            if current_dls > best_dls:
                best_dls = current_dls
                torch.save(model.state_dict(), best_model_path)
                print(f"*** New best model saved: DLS {best_dls:.4f} ***")
            else:
                print(f"DLS {current_dls:.4f} (Best: {best_dls:.4f})")

        # --- Checkpointing ---
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            chk_path = os.path.join(checkpoint_dir, f"bart_sutran_epoch_{epoch+1}.pth")
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_dls': best_dls, # Save best DLS found so far
                'loss': avg_loss # Save avg loss for this epoch
            }
            torch.save(save_dict, chk_path)
            print(f"Checkpoint saved: {chk_path}")

        # --- LR Scheduler Step ---
        if lr_scheduler:
             lr_scheduler.step() # Step based on epoch

    print(f"\n--- Training Complete. Best DLS: {best_dls:.4f} ---")
    return best_dls


# --- Main Execution Block ---
if __name__ == "__main__":
    start_run_time = time.time()
    if not os.path.exists(PROCESSED_TENSOR_PATH) or not os.path.exists(METADATA_PATH):
        print(f"Error: Preprocessed data/metadata missing.")
        print(f"Ensure '{PROCESSED_TENSOR_PATH}' and '{METADATA_PATH}' exist.")
        print(f"These should be generated by the modified 'processdata_sutran_like.py'.")
        exit(1)

    print("Loading data and metadata...")
    try:
        # Load the dictionary containing the tensor tuples
        tensor_data_dict = torch.load(PROCESSED_TENSOR_PATH)
        # Load metadata (assuming it's pickled)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        print("Data and metadata loaded.")
        # Example: Access training data tuple
        train_data_tuple = tensor_data_dict['train']
        val_data_tuple = tensor_data_dict['val']
        test_data_tuple = tensor_data_dict['test'] # Load test tuple for final eval
        print(f"Train data tuple length: {len(train_data_tuple)}")
        print(f"Validation data tuple length: {len(val_data_tuple)}")
        print(f"Test data tuple length: {len(test_data_tuple)}")

    except Exception as e:
        print(f"Error loading data files: {e}")
        exit(1)

    # --- Create Datasets & DataLoaders using TensorDataset ---
    print("Creating Datasets and DataLoaders...")
    try:
        # Use TensorDataset directly with the loaded tuples
        train_ds = TensorDataset(*train_data_tuple)
        val_ds = TensorDataset(*val_data_tuple)
        test_ds = TensorDataset(*test_data_tuple) # Dataset for final testing

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True) # Loader for final eval
        print(f"Loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error creating TensorDataset/DataLoader: {e}")
        print("Check if the loaded tensor tuples have consistent first dimensions.")
        exit(1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Seed ---
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

    # --- Instantiate Model, Optimizer, Loss ---
    model = AdaptedBartModel(metadata, D_MODEL, N_HEADS, FFN_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT).to(device)
    print(f"AdaptedBartModel instantiated ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params).")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn = SutranLikeLossTuple(metadata).to(device) # Use the tuple-adapted loss
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    # LR Scheduler (optional, example using author's choice)
    decay_factor = 0.96
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)
    lr_scheduler_present = True # Set to False if not using scheduler

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_dls = -1.0
    best_mae_rrt = 1e9 # Initialize RRT tracking if needed
    if LOAD_CHECKPOINT_PATH and os.path.isfile(LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint: {LOAD_CHECKPOINT_PATH}")
        try:
            chkpt = torch.load(LOAD_CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(chkpt['model_state_dict'])
            print("Model state loaded.")
            if 'optimizer_state_dict' in chkpt and optimizer:
                try: optimizer.load_state_dict(chkpt['optimizer_state_dict']); print("Optimizer loaded.")
                except: print("Warn: Optimizer load failed.")
            if 'scaler_state_dict' in chkpt and scaler and use_amp:
                scaler.load_state_dict(chkpt['scaler_state_dict']); print("Scaler loaded.")
            start_epoch = chkpt.get('epoch', 0)
            best_dls = chkpt.get('best_dls', -1.0) # Load previous best DLS
            best_mae_rrt = chkpt.get('best_MAE_rrt', 1e9) # Load previous best RRT
            print(f"Resuming from end of epoch {start_epoch}. Best DLS: {best_dls:.4f}")
        except Exception as e: print(f"Error loading checkpoint: {e}. Starting fresh."); start_epoch = 0; best_dls = -1.0; best_mae_rrt = 1e9
    else: print("Starting training from scratch.")


    # --- Run Training ---
    run_training_tuple(model, train_loader, val_loader, optimizer, lr_scheduler if lr_scheduler_present else None, loss_fn, device, EPOCHS, metadata,
                 CHECKPOINT_DIR, BEST_MODEL_PATH, start_epoch, best_dls, SAVE_CHECKPOINT_EVERY, EVALUATE_EVERY, GRADIENT_CLIP_NORM, USE_AMP)

    end_run_time = time.time()
    print(f"--- Training Script Finished ({ (end_run_time - start_run_time)/60:.2f} minutes) ---")

    # --- Optional: Final Evaluation on Test Set ---
    print("\n--- Running Final Evaluation on Test Set ---")
    # Load best model
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model file not found. Evaluating last state.")

    # Use the tuple-adapted evaluation function with the test loader
    final_dls, final_mae_dict = evaluate_model_tuple(model, test_loader, device, loss_fn, metadata)
    print("\n--- Final Test Set Metrics ---")
    print(f"Activity DLS: {final_dls:.4f}")
    for label, mae in final_mae_dict.items():
        print(f"{label} MAE (min): {mae:.4f}")