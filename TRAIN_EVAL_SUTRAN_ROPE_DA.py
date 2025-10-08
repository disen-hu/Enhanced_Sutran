"""Pipeline to train and evaluate SuTraN_RoPE (RoPE on Q/K).
This script mirrors TRAIN_EVAL_SUTRAN_DA.py with minimal changes:
- imports the RoPE-based model
- stores results under SUTRAN_ROPE_DA_results
"""
from ast import main
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle
import sys

# 确保可以导入原仓库中的 SuTraN 代码
# PROJECT_ROOT = os.path.join(os.path.dirname(__file__), 'autodl-tmp', 'sutran', 'SuffixTransformerNetwork-main')
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)


def load_checkpoint(model, path_to_checkpoint, train_or_eval, lr):
    """Loads model checkpoint (weights and optimizer state)."""
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Align with original SuTraN checkpoint keys
    final_epoch_trained = checkpoint.get('epoch:', checkpoint.get('epoch', 0))
    final_loss = checkpoint.get('loss', None)
    if train_or_eval == 'train':
        model.train()
    elif train_or_eval == 'eval':
        model.eval()
    return model, optimizer, final_epoch_trained, final_loss


def train_eval(log_name):
    """Training and evaluating SuTraN_RoPE with paper-like params."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Support both folder name and absolute dataset path
    dataset_base = os.path.basename(log_name)

    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            loaded_dict = pickle.load(file)
        return loaded_dict

    # Load metadata dicts (match original SuTraN script)
    temp_string = dataset_base + '_cardin_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_dict = load_dict(temp_path)
    num_activities = cardinality_dict['concept:name'] + 2

    temp_string = dataset_base + '_cardin_list_prefix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_prefix = load_dict(temp_path)

    temp_string = dataset_base + '_cardin_list_suffix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_suffix = load_dict(temp_path)

    temp_string = dataset_base + '_num_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    num_cols_dict = load_dict(temp_path)

    temp_string = dataset_base + '_cat_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cat_cols_dict = load_dict(temp_path)

    temp_string = dataset_base + '_train_means_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_means_dict = load_dict(temp_path)

    temp_string = dataset_base + '_train_std_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_std_dict = load_dict(temp_path)

    mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
    mean_std_tsp = [train_means_dict['suffix_df'][1], train_std_dict['suffix_df'][1]]
    mean_std_tss = [train_means_dict['suffix_df'][0], train_std_dict['suffix_df'][0]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])
    num_categoricals_pref, num_categoricals_suf = len(cat_cols_dict['prefix_df']), len(cat_cols_dict['suffix_df'])

    d_model = 32 
    num_prefix_encoder_layers = 4
    num_decoder_layers = 4
    num_heads = 8 
    d_ff = 4*d_model 
    layernorm_embeds = True
    outcome_bool = False
    dropout = 0.1
    batch_size = 512

    backup_path = os.path.join(log_name, "SUTRAN_ROPE_DA_results")
    os.makedirs(backup_path, exist_ok=True)

    # Load datasets (.pt format like original scripts)
    train_dataset = torch.load(os.path.join(log_name, 'train_tensordataset.pt'))
    val_dataset = torch.load(os.path.join(log_name, 'val_tensordataset.pt'))
    test_dataset = torch.load(os.path.join(log_name, 'test_tensordataset.pt'))

    # Create TensorDataset for training
    train_dataset = TensorDataset(*train_dataset)

    # Model
    from SuTraN_RoPE.SuTraN_RoPE import SuTraN_RoPE
    model = SuTraN_RoPE(num_activities=num_activities, 
                        d_model=d_model, 
                        cardinality_categoricals_pref=cardinality_list_prefix, 
                        num_numericals_pref=num_numericals_pref, 
                        num_prefix_encoder_layers=num_prefix_encoder_layers, 
                        num_decoder_layers=num_decoder_layers, 
                        num_heads=num_heads, 
                        d_ff=4*d_model, 
                        dropout=dropout, 
                        remaining_runtime_head=True, 
                        layernorm_embeds=layernorm_embeds, 
                        outcome_bool=outcome_bool)
    model.to(device)

    # Optimizer & LR scheduler
    decay_factor = 0.96
    lr = 0.0002
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)

    # Training procedure (reuse original)
    from SuTraN.train_procedure import train_model
    start_epoch = 0
    num_epochs = 20
    num_classes = num_activities 
    batch_interval = 800
    train_model(model, 
                optimizer, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                True,
                outcome_bool,
                num_classes, 
                batch_interval, 
                backup_path, 
                num_categoricals_pref, 
                mean_std_ttne, 
                mean_std_tsp, 
                mean_std_tss, 
                mean_std_rrt, 
                batch_size, 
                patience=24,
                lr_scheduler_present=True, 
                lr_scheduler=lr_scheduler)

    # Load best checkpoint for final inference
    import pandas as pd
    final_results_path = os.path.join(backup_path, 'backup_results.csv')
    df = pd.read_csv(final_results_path)
    dl_col = 'Activity suffix: 1-DL (validation)'
    rrt_col = 'RRT - mintues MAE validation'
    df['rrt_rank_val'] = df[rrt_col].rank(method='min').astype(int)
    df['dl_rank_val'] = df[dl_col].rank(method='min', ascending=False).astype(int)
    df['summed_rank_val'] = df['rrt_rank_val'] + df['dl_rank_val']
    row_with_lowest_loss = df.loc[df['summed_rank_val'].idxmin()]
    epoch_value = row_with_lowest_loss['epoch']
    best_epoch_string = 'model_epoch_{}.pt'.format(int(epoch_value))
    best_epoch_path = os.path.join(backup_path, best_epoch_string)

    model = SuTraN_RoPE(num_activities=num_activities, 
                        d_model=d_model, 
                        cardinality_categoricals_pref=cardinality_list_prefix, 
                        num_numericals_pref=num_numericals_pref, 
                        num_prefix_encoder_layers=num_prefix_encoder_layers, 
                        num_decoder_layers=num_decoder_layers, 
                        num_heads=num_heads, 
                        d_ff=4*d_model, 
                        dropout=dropout, 
                        remaining_runtime_head=True, 
                        layernorm_embeds=layernorm_embeds, 
                        outcome_bool=outcome_bool)
    model.to(device)

    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.eval()

    # Inference
    from SuTraN.inference_procedure import inference_loop
    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)
    inf_results = inference_loop(model, 
                                 test_dataset, 
                                 True, 
                                 outcome_bool, 
                                 num_categoricals_pref, 
                                 mean_std_ttne, 
                                 mean_std_tsp, 
                                 mean_std_tss, 
                                 mean_std_rrt, 
                                 results_path=results_path, 
                                 val_batch_size=512)

    avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
    avg_dam_lev = inf_results[2]
    perc_too_early = inf_results[3]
    perc_too_late = inf_results[4]
    perc_correct = inf_results[5]
    avg_MAE_stand_RRT = inf_results[9]
    avg_MAE_minutes_RRT = inf_results[10]

    print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
    print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
    print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; too late - {}".format(perc_too_early, perc_correct, perc_too_late))
    print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))
if __name__ == "__main__":
    # 指定要训练和评估的数据集的名称（数据位于原仓库内）
    dataset_log_name = 'BPIC_17_DR'
    print(f"Starting training and evaluation for dataset: {dataset_log_name}")

    train_eval(log_name=dataset_log_name)
    print(f"Finished training and evaluation for dataset: {dataset_log_name}")
    print(f"Results saved in: {os.path.join(dataset_log_name, 'SUTRAN_ROPE_DA_results')}\n")