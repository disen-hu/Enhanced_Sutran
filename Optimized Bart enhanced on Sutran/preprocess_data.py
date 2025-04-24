# -*- coding: utf-8 -*-
# processdata_aligned_with_sutran.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset # 将使用这个
import os
import pickle
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Configuration - 根据您的数据进行调整
# =============================================================================
INPUT_CSV_PATH = 'BPIC19.csv' # <--- 提供原始、未分割的日志文件路径
LOG_NAME = 'BPIC_19_processed' # <--- 输出文件夹和文件名的基础
OUTPUT_DIR = LOG_NAME # 输出将保存在以此命名的文件夹中

# 日志过滤参数 (示例值，来自作者的 BPIC19 脚本)
START_DATE = "2018-01"
START_BEFORE_DATE = "2018-09" # 仅用于 BPIC19 的 'workaround' 模式
END_DATE = "2019-02"
MAX_DAYS = 143.33 # 案例最大持续时间（天）
WINDOW_SIZE = 17 # 案例最大长度（事件数）
TEST_LEN_SHARE = 0.25 # 测试集比例
VAL_LEN_SHARE = 0.2 # 验证集比例 (从训练集中划分)
MODE = 'workaround' # 'preferred' 或 'workaround' (BPIC19 使用 'workaround')

# 列名
CASE_ID_COL = 'case:concept:name'
ACTIVITY_COL = 'concept:name'
TIMESTAMP_COL = 'time:timestamp'

# 特征列 - 根据您的 BPIC19.csv 列进行调整
CAT_CASE_FTS = ['case:Spend area text', 'case:Company', 'case:Document Type',
                'case:Sub spend area text', 'case:Item',
                'case:Vendor', 'case:Item Type',
                'case:Item Category', 'case:Spend classification text',
                'case:GR-Based Inv. Verif.', 'case:Goods Receipt']
NUM_CASE_FTS = []
CAT_EVENT_FTS = ['org:resource']
NUM_EVENT_FTS = ['Cumulative net worth (EUR)']
OUTCOME_COL = None # 如果没有结果列，则为 None

# --- 常量 (来自作者代码) ---
PAD_TOKEN_IDX = 0 # 填充通常映射到 0
MISSING_CAT_TOKEN = 'MISSINGVL'
OOV_CAT_TOKEN = 'OOV' # Out-of-Vocabulary
END_TOKEN_LABEL = 'END_TOKEN' # 用于活动标签

# =============================================================================
# Helper Functions (Adapted from SuffixTransformerNetwork/Preprocessing)
# =============================================================================

# --- 从 create_benchmarks.py 改编 ---
def sort_log_internal(df, case_id, timestamp):
    df.loc[:, timestamp] = pd.to_datetime(df[timestamp], utc=True) # 确保 UTC
    df_help = df.sort_values([case_id, timestamp], ascending=[True, True], kind='mergesort').copy()
    df_first = df_help.drop_duplicates(subset=case_id)[[case_id, timestamp]].copy()
    df_first = df_first.sort_values(timestamp, ascending=True, kind='mergesort')
    df_first['case_id_int'] = range(len(df_first))
    df_first = df_first.drop(timestamp, axis=1)
    df = df.merge(df_first, on=case_id, how='left')
    df = df.sort_values(['case_id_int', timestamp], ascending=[True, True], kind='mergesort')
    df = df.drop('case_id_int', axis=1)
    return df.reset_index(drop=True)

def filter_by_date(dataset, start_date, end_date, start_before_date, case_id, timestamp):
    """Applies date filtering."""

    dataset[timestamp] = pd.to_datetime(dataset[timestamp], utc=True)
    if start_date:
        case_starts_df = pd.DataFrame(dataset.groupby(case_id)[timestamp].min().reset_index())
        case_starts_df['date'] = case_starts_df[timestamp].dt.to_period('M')
        cases_after = case_starts_df[case_starts_df['date'].astype('str') >= start_date][case_id].values
        dataset = dataset[dataset[case_id].isin(cases_after)]
    if end_date:
        case_stops_df = pd.DataFrame(dataset.groupby(case_id)[timestamp].max().reset_index())
        case_stops_df['date'] = case_stops_df[timestamp].dt.to_period('M')
        cases_before = case_stops_df[case_stops_df['date'].astype('str') <= end_date][case_id].values
        dataset = dataset[dataset[case_id].isin(cases_before)]
    if start_before_date: # Specific for workaround mode
        case_starts_df = pd.DataFrame(dataset.groupby(case_id)[timestamp].min().reset_index())
        case_starts_df['date'] = case_starts_df[timestamp].dt.to_period('M')
        cases_before = case_starts_df[case_starts_df['date'].astype('str') <= start_before_date][case_id].values
        dataset = dataset[dataset[case_id].isin(cases_before)]
    return dataset.reset_index(drop=True)

def filter_by_duration(dataset, max_duration_days, case_id, timestamp):
    """Filters cases longer than max_duration_days."""
    agg_dict = {timestamp: ['min', 'max']}
    duration_df = pd.DataFrame(dataset.groupby(case_id).agg(agg_dict)).reset_index()
    duration_df["duration_days"] = (duration_df[(timestamp, "max")] - duration_df[(timestamp, "min")]).dt.total_seconds() / (24 * 60 * 60)
    condition_1 = duration_df["duration_days"] <= max_duration_days * 1.00000000001
    cases_retained = duration_df[condition_1][case_id].values
    dataset = dataset[dataset[case_id].isin(cases_retained)].reset_index(drop=True)
    return dataset

def train_test_split_temporal(df, test_len_share, case_id, timestamp, mode):
    """Performs the temporal train/test split with leakage prevention."""
    case_starts_df = df.groupby(case_id)[timestamp].min()
    case_stops_df = df.groupby(case_id)[timestamp].max().to_frame()
    case_nr_list_start = case_starts_df.sort_values().index.array
    first_test_case_idx = int(len(case_nr_list_start) * (1 - test_len_share))
    first_test_start_time = np.sort(case_starts_df.values)[first_test_case_idx]

    prefix_dict = {} # Stores info about overlapping cases

    if mode == 'preferred':
        test_case_ids_all = list(case_stops_df[case_stops_df[timestamp].values >= first_test_start_time].index)
        test_case_ids_sa = list(case_nr_list_start[first_test_case_idx:])
        test_case_ids_overlap = list(set(test_case_ids_all) - set(test_case_ids_sa))
        df_test = df[df[case_id].isin(test_case_ids_all)].reset_index(drop=True).copy()
        train_case_ids = case_stops_df[case_stops_df[timestamp].values < first_test_start_time].index.array
        df_train = df[df[case_id].isin(train_case_ids)].reset_index(drop=True).copy()

        # Calculate first event index after split for overlapping test cases
        if test_case_ids_overlap:
            df_test_overlap = df_test[df_test[case_id].isin(test_case_ids_overlap)].copy()
            df_test_overlap['evt_idx'] = df_test_overlap.groupby([case_id]).cumcount()
            df_test_overlap_prefixes = df_test_overlap[df_test_overlap[timestamp].values > first_test_start_time].copy()
            df_test_overlap_prefixes = df_test_overlap_prefixes.groupby(case_id, sort=False, as_index=False).first()
            prefix_dict = df_test_overlap_prefixes.set_index(case_id)['evt_idx'].to_dict() # first_prefix_dict

    elif mode == 'workaround':
        test_case_ids = list(case_starts_df[case_starts_df.values >= first_test_start_time].index.array)
        df_test = df[df[case_id].isin(test_case_ids)].copy().reset_index(drop=True)
        train_case_ids_all = list(case_starts_df[case_starts_df.values < first_test_start_time].index.array)
        df_train = df[df[case_id].isin(train_case_ids_all)].copy().reset_index(drop=True)

        # Calculate last event index before split for overlapping train cases
        train_case_ids_eb = list(case_stops_df[case_stops_df[timestamp].values < first_test_start_time].index.array)
        train_case_ids_overlap = list(set(train_case_ids_all) - set(train_case_ids_eb))
        if train_case_ids_overlap:
            df_train_overlap = df_train[df_train[case_id].isin(train_case_ids_overlap)].copy()
            df_train_overlap['evt_idx'] = df_train_overlap.groupby([case_id]).cumcount()
            df_train_overlap_prefixes = df_train_overlap[df_train_overlap[timestamp].values < first_test_start_time].copy()
            df_train_overlap_prefixes = df_train_overlap_prefixes.groupby(case_id, sort=False, as_index=False).last()
            prefix_dict = df_train_overlap_prefixes.set_index(case_id)['evt_idx'].to_dict() # last_prefix_dict
    else:
        raise ValueError("mode must be 'preferred' or 'workaround'")

    return df_train, df_test, prefix_dict

# --- 从 dataframes_pipeline.py 改编 ---
def split_train_val_internal(df, val_len_share, case_id, timestamp):
    """Splits training data into train and validation sets temporally."""
    case_start_df = df.groupby(case_id)[timestamp].min().reset_index().sort_values(by=timestamp)
    ordered_id_list = list(case_start_df[case_id])
    first_val_case_idx = int(len(ordered_id_list) * (1 - val_len_share))
    val_case_ids = ordered_id_list[first_val_case_idx:]
    train_case_ids = ordered_id_list[:first_val_case_idx]
    train_set = df[df[case_id].isin(train_case_ids)].copy().reset_index(drop=True)
    val_set = df[df[case_id].isin(val_case_ids)].copy().reset_index(drop=True)
    return train_set, val_set

def create_numeric_timeCols_internal(df, case_id, timestamp):
    """Adds numeric time columns."""
    df.loc[:, 'case_length'] = df.groupby(case_id, sort=False)[timestamp].transform('size')
    df = sort_log_internal(df, case_id, timestamp) # Ensure sorted
    df['last_stamp'] = df.groupby(case_id, sort=False)[timestamp].transform('max')
    df['first_stamp'] = df.groupby(case_id, sort=False)[timestamp].transform('min')
    df['next_stamp'] = df.groupby(case_id, sort=False)[timestamp].shift(-1)
    df['prev_stamp'] = df.groupby(case_id, sort=False)[timestamp].shift(1)

    df['tt_next'] = (df['next_stamp'] - df[timestamp]).dt.total_seconds()
    df['ts_prev'] = (df[timestamp] - df['prev_stamp']).dt.total_seconds()
    df['ts_start'] = (df[timestamp] - df['first_stamp']).dt.total_seconds()
    df['rtime'] = (df['last_stamp'] - df[timestamp]).dt.total_seconds()

    df.drop(['next_stamp', 'prev_stamp', 'first_stamp', 'last_stamp'], axis=1, inplace=True)
    df.fillna({'ts_prev': 0, 'tt_next': 0}, inplace=True) # Fill NaNs for first/last events
    # Clip to ensure non-negative values
    for col in ['ts_prev', 'tt_next', 'rtime', 'ts_start']:
        df[col] = df[col].clip(lower=0)
    return df

# --- 从 categ_mapping_mv.py 改编 ---
def map_categorical_features(train_df, test_df, val_df, cat_cols_ext):
    """Maps categorical features to integers, handling missing and OOV."""
    cardinality_dict = {}
    categorical_mapping_dict = {}
    df_full = pd.concat([train_df, test_df, val_df], ignore_index=True) # Check missing on full data

    for cat_col in tqdm(cat_cols_ext, desc="Mapping Categoricals"):
        has_missing = df_full[cat_col].isna().any()
        train_levels = train_df[cat_col].astype(str).unique()
        test_levels = test_df[cat_col].astype(str).unique()
        val_levels = val_df[cat_col].astype(str).unique()

        current_mapping = {}
        current_id = 0

        # Handle Missing Value Token
        if has_missing:
            current_mapping[MISSING_CAT_TOKEN] = current_id
            current_id += 1
            # Fill NaNs temporarily for mapping
            train_df[cat_col] = train_df[cat_col].fillna(MISSING_CAT_TOKEN)
            test_df.loc[:, cat_col] = test_df[cat_col].fillna(MISSING_CAT_TOKEN)
            val_df[cat_col] = val_df[cat_col].fillna(MISSING_CAT_TOKEN)
            # Remove MISSINGVL from levels if present after fillna
            train_levels = [lvl for lvl in train_levels if pd.notna(lvl)]
            test_levels = [lvl for lvl in test_levels if pd.notna(lvl)]
            val_levels = [lvl for lvl in val_levels if pd.notna(lvl)]

        # Map training levels
        for level in sorted(train_levels):
            if level != MISSING_CAT_TOKEN and level not in current_mapping:
                current_mapping[level] = current_id
                current_id += 1

        # Handle OOV Token
        has_oov = any(lvl not in current_mapping for lvl in test_levels) or \
                  any(lvl not in current_mapping for lvl in val_levels)
        if has_oov:
            oov_id = current_id
            current_mapping[OOV_CAT_TOKEN] = oov_id
            current_id += 1
        else:
             oov_id = -1 # Sentinel value if no OOV needed

        # Apply mapping
        train_df[cat_col] = train_df[cat_col].astype(str).map(current_mapping).fillna(oov_id if has_oov else current_mapping.get(MISSING_CAT_TOKEN, 0)).astype(int)
        test_df.loc[:, cat_col] = test_df[cat_col].astype(str).map(lambda x: current_mapping.get(x, oov_id)).astype(int)
        val_df[cat_col] = val_df[cat_col].astype(str).map(lambda x: current_mapping.get(x, oov_id)).astype(int)

        cardinality_dict[cat_col] = current_id # Final size of the mapped integers
        categorical_mapping_dict[cat_col] = current_mapping # Store the mapping

    return train_df, test_df, val_df, cardinality_dict, categorical_mapping_dict

# --- 从 prefix_suffix_creation.py 改编 ---
def create_prefix_suffixes_internal(df, window_size, outcome_col, case_id, timestamp, act_label,
                                    cat_casefts, num_casefts, cat_eventfts, num_eventfts):
    """Creates prefix, suffix, and label dataframes."""
    # Define columns needed for each dataframe
    prefix_cols = [case_id, act_label, timestamp, 'ts_start', 'ts_prev', 'case_length'] + cat_casefts + num_casefts + cat_eventfts + num_eventfts
    suffix_cols = [case_id, act_label, 'ts_start', 'ts_prev', 'case_length'] # No timestamp needed for suffix input
    actLabel_cols = [case_id, act_label, 'case_length']
    timeLabel_cols = [case_id, 'tt_next', 'rtime', 'case_length']
    outcomeLabel_cols = [case_id, outcome_col, 'case_length'] if outcome_col else []

    # Subsets
    prefix_subset = df[prefix_cols].copy()
    suffix_subset = df[suffix_cols].copy()
    actLabel_subset = df[actLabel_cols].copy()
    timeLabel_subset = df[timeLabel_cols].copy()
    outcomeLabel_subset = df[outcomeLabel_cols].copy() if outcome_col else pd.DataFrame()

    # Add END_TOKEN to activity labels
    actLabel_subset['evt_idx'] = actLabel_subset.groupby(case_id, sort=False).cumcount()
    case_data = df.drop_duplicates(subset=case_id).reset_index(drop=True)[[case_id, 'case_length']].copy()
    end_df = pd.DataFrame({
        case_id: case_data[case_id],
        act_label: END_TOKEN_LABEL, # Use the constant
        'case_length': case_data['case_length'],
        'evt_idx': case_data['case_length'] # Index after last event
    })
    actLabel_subset = pd.concat([actLabel_subset, end_df], ignore_index=True).sort_values([case_id, 'evt_idx'])
    actLabel_subset = actLabel_subset.drop(columns=['evt_idx'])

    # Generate pairs
    all_prefix_dfs, all_suffix_dfs, all_timeLabel_dfs, all_actLabel_dfs, all_outcomeLabel_dfs = [], [], [], [], []

    unique_cases = df[case_id].unique()
    for case in tqdm(unique_cases, desc="Generating Prefix/Suffix Pairs"):
        case_df = df[df[case_id] == case]
        case_len = len(case_df)

        for k in range(1, case_len + 1): # k is prefix length (1 to n)
            prefix_k_df = prefix_subset[prefix_subset[case_id] == case].head(k).copy()
            suffix_k_df = suffix_subset[suffix_subset[case_id] == case].iloc[k-1:].copy() # Suffix starts from last prefix event
            timeLabel_k_df = timeLabel_subset[timeLabel_subset[case_id] == case].iloc[k-1:].copy()
            actLabel_k_df = actLabel_subset[actLabel_subset[case_id] == case].iloc[k:].copy() # Activity labels start from next event
            outcomeLabel_k_df = outcomeLabel_subset[outcomeLabel_subset[case_id] == case].head(1).copy() if outcome_col else pd.DataFrame()

            # Assign unique ID for this prefix-suffix pair
            pair_id = f"{case}_{k}"
            prefix_k_df[case_id] = pair_id
            suffix_k_df[case_id] = pair_id
            timeLabel_k_df[case_id] = pair_id
            actLabel_k_df[case_id] = pair_id
            if outcome_col: outcomeLabel_k_df[case_id] = pair_id

            all_prefix_dfs.append(prefix_k_df)
            all_suffix_dfs.append(suffix_k_df)
            all_timeLabel_dfs.append(timeLabel_k_df)
            all_actLabel_dfs.append(actLabel_k_df)
            if outcome_col: all_outcomeLabel_dfs.append(outcomeLabel_k_df)

    prefix_df_final = pd.concat(all_prefix_dfs, ignore_index=True)
    suffix_df_final = pd.concat(all_suffix_dfs, ignore_index=True)
    timeLabel_df_final = pd.concat(all_timeLabel_dfs, ignore_index=True)
    actLabel_df_final = pd.concat(all_actLabel_dfs, ignore_index=True)
    outcomeLabel_df_final = pd.concat(all_outcomeLabel_dfs, ignore_index=True) if outcome_col else None

    if outcome_col:
        return prefix_df_final, suffix_df_final, timeLabel_df_final, actLabel_df_final, outcomeLabel_df_final
    else:
        return prefix_df_final, suffix_df_final, timeLabel_df_final, actLabel_df_final

def remove_overlapping_pairs(pref_suff_tuple, outcome_col, case_id_col, prefix_dict, mode):
    """Removes overlapping pairs based on mode and prefix_dict."""
    prefix_df, suffix_df, timeLabel_df, actLabel_df = pref_suff_tuple[:4]
    outcomeLabel_df = pref_suff_tuple[4] if outcome_col else None

    # Extract original case ID and prefix number (k) from the pair ID
    prefix_df[['orig_case_id', 'prefix_nr']] = prefix_df[case_id_col].str.rsplit('_', n=1, expand=True)
    prefix_df['prefix_nr'] = pd.to_numeric(prefix_df['prefix_nr'])

    invalid_pair_ids = set()

    if mode == 'preferred': # Remove test pairs whose prefix ends before the split point
        for case, first_valid_idx in prefix_dict.items():
            # Indices are 0-based, prefix_nr is 1-based length
            # We need to remove pairs where prefix_nr <= first_valid_idx
            # Example: first_valid_idx = 5 (6th event). Remove pairs with prefix_nr 1, 2, 3, 4, 5
            invalid_nrs = range(1, first_valid_idx + 1)
            for nr in invalid_nrs:
                invalid_pair_ids.add(f"{case}_{nr}")
    elif mode == 'workaround': # Remove train pairs whose prefix contains events after the split point
        for case, last_valid_idx in prefix_dict.items():
            # Indices are 0-based, prefix_nr is 1-based length
            # We need to remove pairs where prefix_nr > last_valid_idx + 1
            # Example: last_valid_idx = 5 (6th event). Remove pairs with prefix_nr 7, 8, ...
            max_len = prefix_df.loc[prefix_df['orig_case_id'] == case, 'case_length'].iloc[0] # Find max length for this case
            invalid_nrs = range(last_valid_idx + 2, max_len + 1) # +2 because prefix_nr is 1-based length
            for nr in invalid_nrs:
                invalid_pair_ids.add(f"{case}_{nr}")

    print(f"Removing {len(invalid_pair_ids)} overlapping pairs (mode: {mode})...")
    prefix_df_clean = prefix_df[~prefix_df[case_id_col].isin(invalid_pair_ids)].drop(columns=['orig_case_id', 'prefix_nr'])
    suffix_df_clean = suffix_df[~suffix_df[case_id_col].isin(invalid_pair_ids)]
    timeLabel_df_clean = timeLabel_df[~timeLabel_df[case_id_col].isin(invalid_pair_ids)]
    actLabel_df_clean = actLabel_df[~actLabel_df[case_id_col].isin(invalid_pair_ids)]
    if outcome_col:
        outcomeLabel_df_clean = outcomeLabel_df[~outcomeLabel_df[case_id_col].isin(invalid_pair_ids)]
        return prefix_df_clean, suffix_df_clean, timeLabel_df_clean, actLabel_df_clean, outcomeLabel_df_clean
    else:
        return prefix_df_clean, suffix_df_clean, timeLabel_df_clean, actLabel_df_clean

# --- 从 treat_numericals.py 改编 ---
def standardize_and_handle_nan(train_df, val_df, test_df, num_cols):
    """Standardizes numerical columns based on train set and handles NaNs."""
    scalers = {}
    indicator_cols_added = []
    df_list = [train_df, val_df, test_df]
    df_names = ['train', 'val', 'test']

    print(f"Standardizing numerical columns: {num_cols}")

    for col in num_cols:
        # --- Fit Scaler on Training Data (excluding NaNs) ---
        train_col_nonan = train_df[col].dropna()
        if train_col_nonan.empty:
            print(f"Warning: Column '{col}' in training data is all NaN. Skipping scaling.")
            scalers[col] = None # Mark as not scaled
            # Fill NaNs with 0 in all sets if the training column was all NaN
            for df in df_list:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
            continue # Skip to next column

        scaler = StandardScaler()
        # Reshape for scaler
        scaler.fit(train_col_nonan.values.reshape(-1, 1))
        scalers[col] = scaler

        # --- Apply Scaling and Handle NaNs ---
        indicator_col_name = f"{col}_missing"
        nan_present_anywhere = False
        for df in df_list:
            if col in df.columns and df[col].isna().any():
                nan_present_anywhere = True
                break

        if nan_present_anywhere:
            indicator_cols_added.append(indicator_col_name)
            print(f"  Handling NaNs and adding indicator for: {col}")

        for i, df in enumerate(df_list):
            if col in df.columns:
                col_data = df[col].values.reshape(-1, 1)
                nan_mask = np.isnan(col_data).flatten()

                # Apply scaling only to non-NaN values
                scaled_data = col_data.copy() # Start with original data (including NaNs)
                if np.any(~nan_mask): # Check if there are non-NaN values to scale
                    scaled_data[~nan_mask] = scaler.transform(col_data[~nan_mask].reshape(-1, 1)).flatten()

                # Add indicator column if needed
                if nan_present_anywhere:
                    df[indicator_col_name] = nan_mask.astype(int)

                # Fill NaNs with 0 *after* scaling
                df[col] = np.nan_to_num(scaled_data, nan=0.0).flatten()
            elif nan_present_anywhere: # Ensure indicator column exists even if original col is missing
                 df[indicator_col_name] = 0

    print(f"Added NaN indicator columns: {indicator_cols_added}")
    return train_df, val_df, test_df, scalers, indicator_cols_added

def standardize_time_labels(train_df, val_df, test_df, time_label_cols, case_id_col):
    """Standardizes time label columns based on training set."""
    scalers = {}
    print(f"Standardizing time labels: {time_label_cols}")
    for col in time_label_cols:
        scaler = StandardScaler()
        if col == 'rtime':
            # Fit RRT scaler only on the first RRT value per case in training set
            train_first_rrt = train_df.drop_duplicates(subset=case_id_col, keep='first')[col].dropna()
            if not train_first_rrt.empty:
                scaler.fit(train_first_rrt.values.reshape(-1, 1))
            else:
                print(f"Warning: No non-NaN first 'rtime' values found in training set for scaling.")
                scaler = None # Cannot fit
        else: # tt_next
            train_col_nonan = train_df[col].dropna()
            if not train_col_nonan.empty:
                scaler.fit(train_col_nonan.values.reshape(-1, 1))
            else:
                 print(f"Warning: No non-NaN '{col}' values found in training set for scaling.")
                 scaler = None # Cannot fit

        scalers[col] = scaler

        # Apply transformation to all sets
        for df in [train_df, val_df, test_df]:
            if col in df.columns and scaler is not None:
                # Transform non-NaN values, keep NaNs for now (will be filled later)
                col_data = df[col].values.reshape(-1, 1)
                nan_mask = np.isnan(col_data).flatten()
                scaled_data = col_data.copy()
                if np.any(~nan_mask):
                    scaled_data[~nan_mask] = scaler.transform(col_data[~nan_mask].reshape(-1, 1)).flatten()
                df[col] = scaled_data.flatten() # Keep NaNs after scaling
            elif col in df.columns and scaler is None:
                 df[col] = df[col].fillna(0.0) # Fill with 0 if scaler couldn't be fit

    return train_df, val_df, test_df, scalers

# --- 从 tensor_creation.py 改编 ---
def generate_tensors_from_dfs(pref_suff_tuple, case_id_col, act_label_col, cardinality_dict,
                              num_cols_dict, cat_cols_dict, window_size, outcome_col):
    """Generates the final tuple of tensors from the processed dataframes."""
    prefix_df, suffix_df, timeLabel_df, actLabel_df = pref_suff_tuple[:4]
    outcomeLabel_df = pref_suff_tuple[4] if outcome_col else None

    # Get unique pair IDs and create mapping
    unique_pair_ids = prefix_df[case_id_col].unique()
    pair_id_to_int = {pair_id: i for i, pair_id in enumerate(unique_pair_ids)}
    num_prefs = len(unique_pair_ids)

    # Map pair IDs to integers in all dataframes
    prefix_df['pair_id_int'] = prefix_df[case_id_col].map(pair_id_to_int)
    suffix_df['pair_id_int'] = suffix_df[case_id_col].map(pair_id_to_int)
    timeLabel_df['pair_id_int'] = timeLabel_df[case_id_col].map(pair_id_to_int)
    actLabel_df['pair_id_int'] = actLabel_df[case_id_col].map(pair_id_to_int)
    if outcome_col:
        outcomeLabel_df['pair_id_int'] = outcomeLabel_df[case_id_col].map(pair_id_to_int)

    # --- Generate Prefix Tensors ---
    prefix_tensors = []
    prefix_df['evt_idx'] = prefix_df.groupby('pair_id_int', sort=False).cumcount()
    prefix_idx = torch.from_numpy(prefix_df[['pair_id_int', 'evt_idx']].to_numpy())

    # Categorical Prefix Tensors
    for col in cat_cols_dict['prefix_df']:
        cat_updates = torch.from_numpy(prefix_df[col].to_numpy()).to(torch.long)
        cat_tensor = torch.full((num_prefs, window_size), PAD_TOKEN_IDX, dtype=torch.long) # Right-pad with 0
        # Ensure indices are within bounds before scattering
        valid_mask = (prefix_idx[:, 1] < window_size)
        cat_tensor[prefix_idx[valid_mask, 0], prefix_idx[valid_mask, 1]] = cat_updates[valid_mask]
        prefix_tensors.append(cat_tensor)

    # Numerical Prefix Tensor
    num_cols_pref = num_cols_dict['prefix_df']
    num_pref_tensor = torch.zeros((num_prefs, window_size, len(num_cols_pref)), dtype=torch.float) # Right-pad with 0.0
    if num_cols_pref:
        num_updates = torch.from_numpy(prefix_df[num_cols_pref].to_numpy()).float()
        valid_mask = (prefix_idx[:, 1] < window_size)
        num_pref_tensor[prefix_idx[valid_mask, 0], prefix_idx[valid_mask, 1]] = num_updates[valid_mask]
    prefix_tensors.append(num_pref_tensor)

    # Padding Mask Tensor
    padding_mask = torch.full((num_prefs, window_size), True, dtype=torch.bool) # True where padded
    valid_mask = (prefix_idx[:, 1] < window_size)
    padding_mask[prefix_idx[valid_mask, 0], prefix_idx[valid_mask, 1]] = False
    prefix_tensors.append(padding_mask)

    # --- Generate Suffix Tensors (Decoder Input) ---
    suffix_tensors = []
    suffix_df['evt_idx'] = suffix_df.groupby('pair_id_int', sort=False).cumcount()
    suffix_idx = torch.from_numpy(suffix_df[['pair_id_int', 'evt_idx']].to_numpy())

    # Categorical Suffix Tensor (Activity)
    col = act_label_col # Assuming only activity in suffix input
    cat_updates_suff = torch.from_numpy(suffix_df[col].to_numpy()).to(torch.long)
    cat_suff_tensor = torch.full((num_prefs, window_size), PAD_TOKEN_IDX, dtype=torch.long)
    valid_mask_suff = (suffix_idx[:, 1] < window_size)
    cat_suff_tensor[suffix_idx[valid_mask_suff, 0], suffix_idx[valid_mask_suff, 1]] = cat_updates_suff[valid_mask_suff]
    suffix_tensors.append(cat_suff_tensor)

    # Numerical Suffix Tensor (ts_start, ts_prev)
    num_cols_suff = num_cols_dict['suffix_df']
    num_suff_tensor = torch.zeros((num_prefs, window_size, len(num_cols_suff)), dtype=torch.float)
    if num_cols_suff:
        num_updates_suff = torch.from_numpy(suffix_df[num_cols_suff].to_numpy()).float()
        num_suff_tensor[suffix_idx[valid_mask_suff, 0], suffix_idx[valid_mask_suff, 1]] = num_updates_suff[valid_mask_suff]
    suffix_tensors.append(num_suff_tensor)

    # --- Generate Label Tensors ---
    label_tensors = []
    timeLabel_df['evt_idx'] = timeLabel_df.groupby('pair_id_int', sort=False).cumcount()
    time_idx = torch.from_numpy(timeLabel_df[['pair_id_int', 'evt_idx']].to_numpy())
    valid_mask_time = (time_idx[:, 1] < window_size)

    # Time Labels (tt_next, rtime)
    num_cols_time = num_cols_dict['timeLabel_df']
    time_label_tensor = torch.full((num_prefs, window_size, len(num_cols_time)), -100.0, dtype=torch.float) # Pad with -100
    if num_cols_time:
        time_updates = torch.from_numpy(timeLabel_df[num_cols_time].to_numpy()).float()
        # Fill NaNs resulting from standardization (if scaler was None) with -100
        time_updates = torch.nan_to_num(time_updates, nan=-100.0)
        time_label_tensor[time_idx[valid_mask_time, 0], time_idx[valid_mask_time, 1]] = time_updates[valid_mask_time]
    # Split into separate tt_next and rtime tensors
    label_tensors.append(time_label_tensor[:, :, 0].unsqueeze(-1)) # tt_next
    label_tensors.append(time_label_tensor[:, :, 1].unsqueeze(-1)) # rtime

    # Activity Labels
    actLabel_df['evt_idx'] = actLabel_df.groupby('pair_id_int', sort=False).cumcount()
    act_idx = torch.from_numpy(actLabel_df[['pair_id_int', 'evt_idx']].to_numpy())
    valid_mask_act = (act_idx[:, 1] < window_size)

    act_label_tensor = torch.full((num_prefs, window_size), PAD_TOKEN_IDX, dtype=torch.long) # Pad with 0
    # Map END_TOKEN_LABEL string to its integer ID before creating tensor
    end_token_id = cardinality_dict[act_label_col] + 1 # END token is last ID (+1 for 0-base, +1 for END)
    act_updates_label = actLabel_df[act_label_col].replace(END_TOKEN_LABEL, end_token_id).to_numpy()
    act_updates_label = torch.from_numpy(act_updates_label).to(torch.long)
    act_label_tensor[act_idx[valid_mask_act, 0], act_idx[valid_mask_act, 1]] = act_updates_label[valid_mask_act]
    label_tensors.append(act_label_tensor)

    # Outcome Labels (if applicable)
    outcome_tensor = None
    if outcome_col:
        # Outcome is per-case, so just need to map pair_id_int to the outcome value
        outcome_map = outcomeLabel_df.drop_duplicates('pair_id_int').set_index('pair_id_int')[outcome_col]
        outcome_values = outcome_map.reindex(range(num_prefs)).values
        outcome_tensor = torch.from_numpy(outcome_values).float().unsqueeze(-1) # Shape (num_prefs, 1)
        label_tensors.append(outcome_tensor)

    all_tensors = tuple(prefix_tensors) + tuple(suffix_tensors) + tuple(label_tensors)
    return all_tensors

# =============================================================================
# Main Pipeline Function
# =============================================================================
def run_aligned_preprocessing(
    input_csv_path, output_dir, log_name,
    start_date, start_before_date, end_date, max_days, window_size,
    test_len_share, val_len_share, mode,
    case_id_col, act_label_col, timestamp_col,
    cat_casefts, num_casefts, cat_eventfts, num_eventfts, outcome_col
):
    """Runs the full preprocessing pipeline aligned with Sutran's logic."""
    print(f"--- Starting Aligned Preprocessing for {log_name} ---")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Loaded raw data: {df.shape[0]} events")
    except FileNotFoundError:
        print(f"Error: Input CSV not found at {input_csv_path}"); return

    # 2. Initial Filtering & Sorting
    df = sort_log_internal(df, case_id_col, timestamp_col)
    df = filter_by_date(df, start_date, end_date, start_before_date, case_id_col, timestamp_col)
    df.drop_duplicates(inplace=True, ignore_index=True)
    df = filter_by_duration(df, max_days, case_id_col, timestamp_col)
    print(f"Filtered data: {df.shape[0]} events")

    # 3. Train/Test Split
    df_train_full, df_test, prefix_dict = train_test_split_temporal(
        df, test_len_share, case_id_col, timestamp_col, mode
    )
    print(f"Split: Train {df_train_full.shape[0]} events, Test {df_test.shape[0]} events")

    # 4. Train/Validation Split
    df_train, df_val = split_train_val_internal(
        df_train_full, val_len_share, case_id_col, timestamp_col
    )
    print(f"Split: Final Train {df_train.shape[0]} events, Validation {df_val.shape[0]} events")

    # 5. Filter by Window Size & Select Columns
    all_cat_features = cat_casefts + cat_eventfts
    all_num_features = num_casefts + num_eventfts
    needed_cols = [case_id_col, act_label_col, timestamp_col] + all_cat_features + all_num_features
    if outcome_col: needed_cols.append(outcome_col)

    dfs = {'train': df_train, 'val': df_val, 'test': df_test}
    for name, d in dfs.items():
        d['case_length'] = d.groupby(case_id_col, sort=False)[act_label_col].transform('size')
        d_filtered = d[d['case_length'] <= window_size].copy()
        # Ensure all needed columns exist, add NaN if not (except case_length)
        for col in needed_cols:
            if col not in d_filtered.columns and col != 'case_length':
                d_filtered[col] = np.nan
        dfs[name] = d_filtered[needed_cols + ['case_length']] # Keep case_length for now
        print(f"Filtered {name} by window size ({window_size}): {dfs[name].shape[0]} events")

    # 6. Map Categorical Features
    cat_cols_ext = all_cat_features + [act_label_col]
    df_train, df_test, df_val, cardinality_dict, categorical_mapping_dict = map_categorical_features(
        dfs['train'], dfs['test'], dfs['val'], cat_cols_ext
    )

    # 7. Add Time Features
    print("Adding time features...")
    df_train = create_numeric_timeCols_internal(df_train, case_id_col, timestamp_col)
    df_val = create_numeric_timeCols_internal(df_val, case_id_col, timestamp_col)
    df_test = create_numeric_timeCols_internal(df_test, case_id_col, timestamp_col)

    # 8. Generate Prefix/Suffix DataFrames
    print("Generating prefix/suffix pairs...")
    train_pref_suff = create_prefix_suffixes_internal(
        df_train, window_size, outcome_col, case_id_col, timestamp_col, act_label_col,
        cat_casefts, num_casefts, cat_eventfts, num_eventfts
    )
    val_pref_suff = create_prefix_suffixes_internal(
        df_val, window_size, outcome_col, case_id_col, timestamp_col, act_label_col,
        cat_casefts, num_casefts, cat_eventfts, num_eventfts
    )
    test_pref_suff = create_prefix_suffixes_internal(
        df_test, window_size, outcome_col, case_id_col, timestamp_col, act_label_col,
        cat_casefts, num_casefts, cat_eventfts, num_eventfts
    )

    # 9. Remove Overlapping Pairs (if necessary based on mode)
    if mode == 'preferred':
        test_pref_suff = remove_overlapping_pairs(test_pref_suff, outcome_col, case_id_col, prefix_dict, mode)
    elif mode == 'workaround':
        train_pref_suff = remove_overlapping_pairs(train_pref_suff, outcome_col, case_id_col, prefix_dict, mode)
        val_pref_suff = remove_overlapping_pairs(val_pref_suff, outcome_col, case_id_col, prefix_dict, mode)

    # 10. Standardize Numerical Features & Handle NaNs
    print("Standardizing numerical features and handling NaNs...")
    # Define numerical columns for each part
    pref_numcols_base = num_casefts + num_eventfts + ['ts_start', 'ts_prev']
    suff_numcols = ['ts_start', 'ts_prev'] # Only these for suffix input
    timelab_numcols = ['tt_next', 'rtime']

    # Standardize prefix numericals
    train_pref_suff[0], val_pref_suff[0], test_pref_suff[0], pref_scalers, indicator_cols = standardize_and_handle_nan(
        train_pref_suff[0], val_pref_suff[0], test_pref_suff[0], pref_numcols_base
    )
    pref_numcols_final = pref_numcols_base + indicator_cols # Update list

    # Standardize suffix numericals
    train_pref_suff[1], val_pref_suff[1], test_pref_suff[1], suff_scalers, _ = standardize_and_handle_nan(
        train_pref_suff[1], val_pref_suff[1], test_pref_suff[1], suff_numcols
    ) # No indicator cols expected/needed for suffix inputs usually

    # Standardize time labels
    train_pref_suff[2], val_pref_suff[2], test_pref_suff[2], time_label_scalers = standardize_time_labels(
         train_pref_suff[2], val_pref_suff[2], test_pref_suff[2], timelab_numcols, case_id_col
    )
    # Fill remaining NaNs in time labels (e.g., from standardization failure) with -100 padding value
    for i in range(2, 3): # Index 2 is timeLabel_df
        train_pref_suff[i].fillna(-100.0, inplace=True)
        val_pref_suff[i].fillna(-100.0, inplace=True)
        test_pref_suff[i].fillna(-100.0, inplace=True)


    # 11. Prepare Final Dictionaries for Tensor Creation & Saving
    num_cols_dict = {'prefix_df': pref_numcols_final, 'suffix_df': suff_numcols, 'timeLabel_df': timelab_numcols}
    cat_cols_dict = {'prefix_df': cat_casefts + cat_eventfts + [act_label_col],
                     'suffix_df': [act_label_col], # Only activity needed for suffix input usually
                     'actLabel_df': [act_label_col]} # Activity label target
    all_scalers = {'prefix': pref_scalers, 'suffix': suff_scalers, 'timeLabel': time_label_scalers}

    # Extract means and stds from scalers for saving (more robust than pickling scalers)
    train_means_dict = {key: [s.mean_[0] if s else 0.0 for s in scalers.values()] for key, scalers in all_scalers.items()}
    train_std_dict = {key: [np.sqrt(s.var_[0]) if s and s.var_[0] > 0 else 1.0 for s in scalers.values()] for key, scalers in all_scalers.items()}
    # Add means/stds for the specific time labels separately for easier access later
    train_means_dict['timeLabel_df'] = [time_label_scalers['tt_next'].mean_[0] if time_label_scalers.get('tt_next') else 0.0,
                                        time_label_scalers['rtime'].mean_[0] if time_label_scalers.get('rtime') else 0.0]
    train_std_dict['timeLabel_df'] = [np.sqrt(time_label_scalers['tt_next'].var_[0]) if time_label_scalers.get('tt_next') and time_label_scalers['tt_next'].var_[0] > 0 else 1.0,
                                      np.sqrt(time_label_scalers['rtime'].var_[0]) if time_label_scalers.get('rtime') and time_label_scalers['rtime'].var_[0] > 0 else 1.0]
    # Add means/stds for suffix ts_start and ts_prev
    train_means_dict['suffix_df'] = [suff_scalers['ts_start'].mean_[0] if suff_scalers.get('ts_start') else 0.0,
                                     suff_scalers['ts_prev'].mean_[0] if suff_scalers.get('ts_prev') else 0.0]
    train_std_dict['suffix_df'] = [np.sqrt(suff_scalers['ts_start'].var_[0]) if suff_scalers.get('ts_start') and suff_scalers['ts_start'].var_[0] > 0 else 1.0,
                                   np.sqrt(suff_scalers['ts_prev'].var_[0]) if suff_scalers.get('ts_prev') and suff_scalers['ts_prev'].var_[0] > 0 else 1.0]


    # 12. Generate Final Tensors
    print("Generating final tensors...")
    train_tensors = generate_tensors_from_dfs(
        train_pref_suff, case_id_col, act_label_col, cardinality_dict,
        num_cols_dict, cat_cols_dict, window_size, outcome_col
    )
    val_tensors = generate_tensors_from_dfs(
        val_pref_suff, case_id_col, act_label_col, cardinality_dict,
        num_cols_dict, cat_cols_dict, window_size, outcome_col
    )
    test_tensors = generate_tensors_from_dfs(
        test_pref_suff, case_id_col, act_label_col, cardinality_dict,
        num_cols_dict, cat_cols_dict, window_size, outcome_col
    )

    # 13. Save Tensors and Metadata
    print("Saving outputs...")
    # --- Save Tensors ---
    torch.save(train_tensors, os.path.join(output_dir, 'train_tensordataset.pt'))
    torch.save(val_tensors, os.path.join(output_dir, 'val_tensordataset.pt'))
    torch.save(test_tensors, os.path.join(output_dir, 'test_tensordataset.pt'))
    print("Saved tensor datasets.")

    # --- Save Metadata ---
    # Cardinality dict (integers)
    with open(os.path.join(output_dir, f'{log_name}_cardin_dict.pkl'), 'wb') as f:
        pickle.dump(cardinality_dict, f)
    # Categorical mapping dict (str -> int)
    with open(os.path.join(output_dir, f'{log_name}_categ_mapping.pkl'), 'wb') as f:
        pickle.dump(categorical_mapping_dict, f)
    # Numerical column names dict
    with open(os.path.join(output_dir, f'{log_name}_num_cols_dict.pkl'), 'wb') as f:
        pickle.dump(num_cols_dict, f)
    # Categorical column names dict
    with open(os.path.join(output_dir, f'{log_name}_cat_cols_dict.pkl'), 'wb') as f:
        pickle.dump(cat_cols_dict, f)
    # Training means dict
    with open(os.path.join(output_dir, f'{log_name}_train_means_dict.pkl'), 'wb') as f:
        pickle.dump(train_means_dict, f)
    # Training std devs dict
    with open(os.path.join(output_dir, f'{log_name}_train_std_dict.pkl'), 'wb') as f:
        pickle.dump(train_std_dict, f)
    # Cardinality lists (needed by some benchmark models)
    cardinality_list_prefix = [cardinality_dict[col] for col in cat_cols_dict['prefix_df']]
    cardinality_list_suffix = [cardinality_dict[col] for col in cat_cols_dict['suffix_df']]
    with open(os.path.join(output_dir, f'{log_name}_cardin_list_prefix.pkl'), 'wb') as f:
        pickle.dump(cardinality_list_prefix, f)
    with open(os.path.join(output_dir, f'{log_name}_cardin_list_suffix.pkl'), 'wb') as f:
        pickle.dump(cardinality_list_suffix, f)

    print("Saved metadata dictionaries and lists.")
    print(f"--- Preprocessing for {log_name} finished. Outputs in '{output_dir}' ---")


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    # --- Preprocess BPIC19 Data ---
    run_aligned_preprocessing(
        input_csv_path=INPUT_CSV_PATH,
        output_dir=OUTPUT_DIR,
        log_name=LOG_NAME,
        start_date=START_DATE,
        start_before_date=START_BEFORE_DATE,
        end_date=END_DATE,
        max_days=MAX_DAYS,
        window_size=WINDOW_SIZE,
        test_len_share=TEST_LEN_SHARE,
        val_len_share=VAL_LEN_SHARE,
        mode=MODE,
        case_id_col=CASE_ID_COL,
        act_label_col=ACTIVITY_COL,
        timestamp_col=TIMESTAMP_COL,
        cat_casefts=CAT_CASE_FTS,
        num_casefts=NUM_CASE_FTS,
        cat_eventfts=CAT_EVENT_FTS,
        num_eventfts=NUM_EVENT_FTS,
        outcome_col=OUTCOME_COL
    )

    # --- Verification (Optional) ---
    print("\n--- Verifying Outputs ---")
    if os.path.exists(OUTPUT_DIR):
        try:
            train_t = torch.load(os.path.join(OUTPUT_DIR, 'train_tensordataset.pt'))
            val_t = torch.load(os.path.join(OUTPUT_DIR, 'val_tensordataset.pt'))
            test_t = torch.load(os.path.join(OUTPUT_DIR, 'test_tensordataset.pt'))
            print(f"Successfully loaded tensor datasets.")
            print(f"  Train tensor tuple length: {len(train_t)}")
            print(f"  Example train tensor shape (prefix activity): {train_t[0].shape}")
            print(f"  Example train tensor shape (prefix numerical): {train_t[len(CAT_CASE_FTS) + len(CAT_EVENT_FTS) + 1].shape}") # Index of numerical tensor
            print(f"  Example train tensor shape (activity labels): {train_t[-1].shape if not OUTCOME_COL else train_t[-2].shape}") # Activity label tensor index

            with open(os.path.join(OUTPUT_DIR, f'{LOG_NAME}_cardin_dict.pkl'), 'rb') as f:
                 cardin = pickle.load(f)
                 print(f"Loaded cardinality dict. Example: {ACTIVITY_COL} cardinality = {cardin.get(ACTIVITY_COL, 'N/A')}")
            with open(os.path.join(OUTPUT_DIR, f'{LOG_NAME}_train_means_dict.pkl'), 'rb') as f:
                 means = pickle.load(f)
                 print(f"Loaded means dict. Example time label means: {means.get('timeLabel_df', 'N/A')}")

        except Exception as e:
            print(f"Error during verification: {e}")
    else:
        print(f"Output directory '{OUTPUT_DIR}' not found.")