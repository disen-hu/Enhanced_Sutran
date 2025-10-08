"""
此模組包含訓練和評估 MambaPPM 的完整流程。
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle

# 假設 load_checkpoint 函數是通用的，我們直接複用
from TRAIN_EVAL_SUTRAN_DA import load_checkpoint

def train_eval(log_name):
    """
    使用 SuTraN 論文中的參數訓練和自動評估 MambaPPM (DA 版本)。
    """
    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            loaded_dict = pickle.load(file)
        return loaded_dict

    # --- 載入資料和設定 (與 TRAIN_EVAL_SUTRAN_DA.py 完全相同) ---
    temp_string = log_name + '_cardin_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_dict = load_dict(temp_path)
    num_activities = cardinality_dict['concept:name'] + 2

    temp_string = log_name + '_cardin_list_prefix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_prefix = load_dict(temp_path)

    temp_string = log_name + '_num_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    num_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_cat_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cat_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_train_means_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_means_dict = load_dict(temp_path)

    temp_string = log_name + '_train_std_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_std_dict = load_dict(temp_path)

    mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
    mean_std_tsp = [train_means_dict['suffix_df'][1], train_std_dict['suffix_df'][1]]
    mean_std_tss = [train_means_dict['suffix_df'][0], train_std_dict['suffix_df'][0]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])
    num_categoricals_pref = len(cat_cols_dict['prefix_df'])
    
    # --- 模型超參數 ---
    d_model = 64  # Mamba 通常使用稍大的 d_model
    n_layers = 4
    d_state = 16
    d_conv = 4
    expand = 2
    layernorm_embeds = True
    outcome_bool = False
    remaining_runtime_head = True
    dropout = 0.2
    batch_size = 128
    
    # 指定結果和回調的路徑
    backup_path = os.path.join(log_name, "MAMBA_DA_results")
    os.makedirs(backup_path, exist_ok=True)

    # 載入資料集
    train_dataset = torch.load(os.path.join(log_name, 'train_tensordataset.pt'))
    val_dataset = torch.load(os.path.join(log_name, 'val_tensordataset.pt'))
    test_dataset = torch.load(os.path.join(log_name, 'test_tensordataset.pt'))
    train_dataset = TensorDataset(*train_dataset)

    # 設定 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # --- 初始化 Mamba 模型 ---
    import random
    torch.manual_seed(24)
    np.random.seed(24)
    random.seed(24)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(24)

    from Mamba.mamba_model import MambaPPM  # 從新檔案匯入模型
    
    model = MambaPPM(
        num_activities=num_activities,
        d_model=d_model,
        n_layers=n_layers,
        cardinality_categoricals_pref=cardinality_list_prefix,
        num_numericals_pref=num_numericals_pref,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        remaining_runtime_head=remaining_runtime_head,
        layernorm_embeds=layernorm_embeds,
        outcome_bool=outcome_bool
    )
    model.to(device)

    # --- 訓練流程 (與 TRAIN_EVAL_SUTRAN_DA.py 相同) ---
    lr = 0.0002
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
    
    from SuTraN.train_procedure import train_model # 複用 SuTraN 的訓練流程
    
    train_model(
        model, optimizer, train_dataset, val_dataset,
        start_epoch=0, num_epochs=200,
        remaining_runtime_head=remaining_runtime_head,
        outcome_bool=outcome_bool,
        num_classes=num_activities,
        batch_interval=800,
        path_name=backup_path,
        num_categoricals_pref=num_categoricals_pref,
        mean_std_ttne=mean_std_ttne,
        mean_std_tsp=mean_std_tsp,
        mean_std_tss=mean_std_tss,
        mean_std_rrt=mean_std_rrt,
        batch_size=batch_size,
        patience=24,
        lr_scheduler_present=True,
        lr_scheduler=lr_scheduler
    )

    # --- 評估流程 (與 TRAIN_EVAL_SUTRAN_DA.py 幾乎相同) ---
    # 重新初始化模型以載入最佳權重
    model = MambaPPM(
        num_activities=num_activities, d_model=d_model, n_layers=n_layers,
        cardinality_categoricals_pref=cardinality_list_prefix,
        num_numericals_pref=num_numericals_pref, d_state=d_state,
        d_conv=d_conv, expand=expand, dropout=dropout,
        remaining_runtime_head=remaining_runtime_head,
        layernorm_embeds=layernorm_embeds, outcome_bool=outcome_bool
    )
    model.to(device)

    final_results_path = os.path.join(backup_path, 'backup_results.csv')
    df = pd.read_csv(final_results_path)
    # ... (尋找最佳 epoch 的邏輯與 SuTraN 相同)
    dl_col = 'Activity suffix: 1-DL (validation)'
    rrt_col = 'RRT - mintues MAE validation'
    df['rrt_rank_val'] = df[rrt_col].rank(method='min').astype(int)
    df['dl_rank_val'] = df[dl_col].rank(method='min', ascending=False).astype(int)
    df['summed_rank_val'] = df['rrt_rank_val'] + df['dl_rank_val']
    row_with_lowest_loss = df.loc[df['summed_rank_val'].idxmin()]
    epoch_value = row_with_lowest_loss['epoch']
    
    best_epoch_path = os.path.join(backup_path, f'model_epoch_{int(epoch_value)}.pt')
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=lr)
    model.eval()

    from SuTraN.inference_procedure import inference_loop # 複用 SuTraN 的推理流程
    
    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)
    
    inf_results = inference_loop(
        model, test_dataset, remaining_runtime_head, outcome_bool,
        num_categoricals_pref, mean_std_ttne, mean_std_tsp, mean_std_tss,
        mean_std_rrt, results_path=results_path, val_batch_size=2048
    )

    # ... (印出和儲存結果的邏輯與 SuTraN 完全相同) ...
    print("--- Mamba Model Final Results ---")
    # (此處省略了與 SuTraN 相同的結果列印代碼)

if __name__ == '__main__':
    # 假設您想訓練 BPIC_17 資料集
    train_eval(log_name='BPIC_17_DR')