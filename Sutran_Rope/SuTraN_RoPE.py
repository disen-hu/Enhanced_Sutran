import torch 
import torch.nn as nn
import math

from SuTraN_RoPE.transformer_prefix_encoder_rope import EncoderLayerRoPE
from SuTraN_RoPE.transformer_suffix_decoder_rope import DecoderLayerRoPE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SuTraN_RoPE(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 cardinality_categoricals_pref, 
                 num_numericals_pref, 
                 num_prefix_encoder_layers = 3, 
                 num_decoder_layers = 2,
                 num_heads=8, 
                 d_ff = 128, 
                 dropout = 0.2, 
                 remaining_runtime_head = True, 
                 layernorm_embeds = True, 
                 outcome_bool = False,
                 decoding_strategy = "greedy",
                 top_p = 0.2,
                 temperature = 1.0,
                 ):
        super(SuTraN_RoPE, self).__init__()

        self.num_activities = num_activities
        self.d_model = d_model
        self.cardinality_categoricals_pref = cardinality_categoricals_pref
        self.num_categoricals_pref = len(self.cardinality_categoricals_pref)
        self.num_numericals_pref = num_numericals_pref
        self.num_prefix_encoder_layers = num_prefix_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.remaining_runtime_head = remaining_runtime_head
        self.layernorm_embeds = layernorm_embeds
        self.outcome_bool = outcome_bool

        # Decoding configuration
        self.decoding_strategy = decoding_strategy
        self.top_p = float(top_p)
        self.temperature = float(temperature)

        # Embedding sizes: same as original SuTraN
        self.embed_sz_categ_pref = [min(600, round(1.6 * n_cat**0.56)) for n_cat in self.cardinality_categoricals_pref[:-1]]
        self.activity_emb_size = min(600, round(1.6 * self.cardinality_categoricals_pref[-1]**0.56))

        self.cat_embeds_pref = nn.ModuleList([
            nn.Embedding(num_embeddings=self.cardinality_categoricals_pref[i]+1, embedding_dim=self.embed_sz_categ_pref[i], padding_idx=0)
            for i in range(self.num_categoricals_pref-1)
        ])
        self.act_emb = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=self.activity_emb_size, padding_idx=0)

        # Initial projection layers
        self.dim_init_prefix = sum(self.embed_sz_categ_pref) + self.activity_emb_size + self.num_numericals_pref
        self.input_embeddings_encoder = nn.Linear(self.dim_init_prefix, self.d_model)

        self.dim_init_suffix = self.activity_emb_size + 2
        self.input_embeddings_decoder = nn.Linear(self.dim_init_suffix, self.d_model)

        # Transformer stacks with RoPE attention
        self.encoder_layers = nn.ModuleList([EncoderLayerRoPE(d_model, num_heads, d_ff, dropout) for _ in range(self.num_prefix_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayerRoPE(d_model, num_heads, d_ff, dropout) for _ in range(self.num_decoder_layers)])

        # Heads
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities)
        self.fc_out_ttne = nn.Linear(self.d_model, 1)
        if self.remaining_runtime_head:
            self.fc_out_rrt = nn.Linear(self.d_model, 1)
        if self.outcome_bool:
            self.fc_out_out = nn.Linear(self.d_model, 1)
            self.sigmoid_out = nn.Sigmoid()

        if self.layernorm_embeds:
            self.norm_enc_embeds = nn.LayerNorm(self.d_model)
            self.norm_dec_embeds = nn.LayerNorm(self.d_model)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.only_rrt = (not self.outcome_bool) & self.remaining_runtime_head
        self.only_out = self.outcome_bool & (not self.remaining_runtime_head)
        self.both_not = (not self.outcome_bool) & (not self.remaining_runtime_head)
        self.both = self.outcome_bool & self.remaining_runtime_head

    def forward(self, inputs, window_size=None, mean_std_ttne=None, mean_std_tsp=None, mean_std_tss=None):
        # Numerical prefix features
        num_ftrs_pref = inputs[(self.num_categoricals_pref-1)+1]
        # Padding mask for prefix events
        padding_mask_input = inputs[(self.num_categoricals_pref-1)+2]
        idx = self.num_categoricals_pref+2
        # Numerical suffix features (tss, tsp)
        num_ftrs_suf = inputs[idx + 1]

        # Categorical prefix embeddings
        cat_emb_pref = self.cat_embeds_pref[0](inputs[0])
        for i in range(1, self.num_categoricals_pref-1):
            cat_emb_pref = torch.cat((cat_emb_pref, self.cat_embeds_pref[i](inputs[i])), dim=-1)
        act_emb_pref = self.act_emb(inputs[self.num_categoricals_pref-1])
        cat_emb_pref = torch.cat((cat_emb_pref, act_emb_pref), dim=-1)

        # Concatenate cat + num and project
        x = torch.cat((cat_emb_pref, num_ftrs_pref), dim=-1)
        x = self.dropout_layer(x)
        x = self.input_embeddings_encoder(x) * math.sqrt(self.d_model)
        if self.layernorm_embeds:
            x = self.norm_enc_embeds(x)

        # Encoder stack
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, padding_mask_input)

        if self.training:
            # Teacher forcing path
            cat_emb_suf = self.act_emb(inputs[idx])
            target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim=-1)
            target_in = self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)
            if self.layernorm_embeds:
                target_in = self.norm_dec_embeds(target_in)

            dec_output = target_in
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, x, padding_mask_input)

            act_probs = self.fc_out_act(dec_output)
            ttne_pred = self.fc_out_ttne(dec_output)

            if self.only_rrt:
                rrt_pred = self.fc_out_rrt(dec_output)
                return act_probs, ttne_pred, rrt_pred
            elif self.only_out:
                out_pred = self.sigmoid_out(self.fc_out_out(dec_output))
                out_pred = out_pred[:, 0, :]
                return act_probs, ttne_pred, out_pred
            elif self.both:
                rrt_pred = self.fc_out_rrt(dec_output)
                out_pred = self.sigmoid_out(self.fc_out_out(dec_output))
                out_pred = out_pred[:, 0, :]
                return act_probs, ttne_pred, rrt_pred, out_pred
            else:
                return act_probs, ttne_pred
        else:
            # Decoding path: greedy (default) or nucleus (top-p) sampling
            act_inputs = inputs[idx]
            batch_size = act_inputs.size(0)
            suffix_acts_decoded = torch.full((batch_size, window_size), 0, dtype=torch.int64, device=device)
            suffix_ttne_preds = torch.full((batch_size, window_size), 0, dtype=torch.float32, device=device)

            use_nucleus = (getattr(self, "decoding_strategy", "greedy") == "nucleus")
            top_p = max(1e-6, min(getattr(self, "top_p", 0.9), 1.0))
            temperature = max(1e-6, getattr(self, "temperature", 1.0))

            for dec_step in range(0, window_size):
                cat_emb_suf = self.act_emb(act_inputs)
                target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim=-1)
                target_in = self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)
                if self.layernorm_embeds:
                    target_in = self.norm_dec_embeds(target_in)

                dec_output = target_in
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, x, padding_mask_input)

                act_logits = self.fc_out_act(dec_output)
                ttne_pred = self.fc_out_ttne(dec_output)

                act_outputs = act_logits[:, dec_step, :]
                ttne_outputs = ttne_pred[:, dec_step, 0]
                suffix_ttne_preds[:, dec_step] = ttne_outputs

                if dec_step == 0:
                    if self.remaining_runtime_head:
                        rrt_pred = self.fc_out_rrt(dec_output)[:, 0, 0]
                    if self.outcome_bool:
                        out_pred = self.sigmoid_out(self.fc_out_out(dec_output))[:, 0, 0]

                act_outputs[:, 0] = -1e9
                if use_nucleus:
                    # Nucleus (top-p) sampling on activity logits
                    logits = act_outputs / temperature
                    probs = torch.softmax(logits, dim=-1)
                    selected_tokens = []
                    for b in range(batch_size):
                        p = probs[b]
                        sorted_probs, sorted_idx = torch.sort(p, descending=True)
                        cumprobs = torch.cumsum(sorted_probs, dim=0)
                        # minimal set whose cumulative probability >= top_p
                        cutoff_len = int(torch.sum(cumprobs <= top_p).item())
                        if cutoff_len == 0:
                            cutoff_len = 1
                        elif cutoff_len < sorted_probs.size(0):
                            cutoff_len = cutoff_len + 1
                        keep_probs = sorted_probs[:cutoff_len]
                        keep_idx = sorted_idx[:cutoff_len]
                        keep_probs = keep_probs / keep_probs.sum()
                        sampled_rel = torch.multinomial(keep_probs, num_samples=1)
                        sampled_token = keep_idx[sampled_rel]
                        selected_tokens.append(sampled_token[0])
                    act_selected = torch.stack(selected_tokens, dim=0)
                else:
                    # Greedy argmax selection
                    act_selected = torch.argmax(act_outputs, dim=-1)
                suffix_acts_decoded[:, dec_step] = act_selected

                if dec_step < (window_size-1):
                    act_suf_updates = act_selected.clone()
                    act_suf_updates = torch.clamp(act_suf_updates, max=self.num_activities-2)
                    act_inputs[:, dec_step+1] = act_suf_updates

                    time_preds_seconds = ttne_outputs*mean_std_ttne[1] + mean_std_ttne[0]
                    time_preds_seconds = torch.clamp(time_preds_seconds, min=0)

                    tss_stand = num_ftrs_suf[:, dec_step, 0].clone()
                    tss_seconds = torch.clamp(tss_stand*mean_std_tss[1] + mean_std_tss[0], min=0)
                    tss_seconds_new = tss_seconds + time_preds_seconds
                    tss_stand_new = (tss_seconds_new - mean_std_tss[0]) / mean_std_tss[1]
                    tsp_stand_new = (time_preds_seconds - mean_std_tsp[0]) / mean_std_tsp[1]
                    new_suffix_timefeats = torch.cat((tss_stand_new.unsqueeze(-1), tsp_stand_new.unsqueeze(-1)), dim=-1)
                    num_ftrs_suf[:, dec_step+1, :] = new_suffix_timefeats

            if self.only_rrt:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred
            elif self.only_out:
                return suffix_acts_decoded, suffix_ttne_preds, out_pred
            elif self.both:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred, out_pred
            else:
                return suffix_acts_decoded, suffix_ttne_preds