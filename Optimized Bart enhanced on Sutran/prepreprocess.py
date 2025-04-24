# processdata_sutran_like.py (Modified)
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import pickle
from sklearn.preprocessing import StandardScaler
import time
# Removed multiprocessing imports as we'll use the author's sequential approach for generation
# from multiprocessing import Pool, cpu_count

# --- Constants ---
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
END_TOKEN = '<end>' # Sutran uses an END token for activity labels
# Removed SPECIAL_TOKENS as it wasn't used in the tensor creation part directly

# --- Configuration ---
# Keep your configuration or adjust as needed
TRAIN_CSV_PATH = 'BPIC_19_train_initial.csv' # Path to train data
TEST_CSV_PATH = 'BPIC_19_test_initial.csv'   # Path to test data
VALIDATION_SPLIT = 0.2  # Percentage of training data to use for validation
OUTPUT_PATH = 'BPIC_19'  # Output folder name
TENSOR_PATH = os.path.join(OUTPUT_PATH, 'processed_tensors_sutran_v2.pt') # Use new name
METADATA_PATH = os.path.join(OUTPUT_PATH, 'metadata_sutran_v2.pt')      # Use new name

# Define columns - ADJUST THESE based on your actual CSV columns
CASE_ID_COL = 'case:concept:name'
ACTIVITY_COL = 'concept:name'
TIMESTAMP_COL = 'time:timestamp'
# Add categorical event and case features if available (Currently empty based on your code)
CAT_CASE_FTS = []
CAT_EVENT_FTS = []
NUM_CASE_FTS = []
NUM_EVENT_FTS = [] # Example: ['Cumulative net worth (EUR)'] - Add yours if needed

# SuTraN processing options
APPLY_LOG_TRANSFORM = False # Author likely didn't log transform standard time features
WINDOW_SIZE = 17  # Max sequence length (like window_size in SuTraN)
# Removed sampling options as we are replicating the author's full prefix generation
# SAMPLE_PREFIXES = False
# MAX_PREFIXES_PER_CASE = 5
MIN_PREFIX_LENGTH = 1
# Removed NUM_WORKERS

# --- Helper Functions (Keep yours or adapt from author as needed) ---

def sort_log(df, case_id, timestamp):
    """Sorts log by case start time, then by timestamp within case."""
    # Assuming timestamp is already datetime
    # df[timestamp] = pd.to_datetime(df[timestamp])
    # Sort by timestamp within case first
    df = df.sort_values([case_id, timestamp], ascending=True, kind='mergesort')
    # Then sort cases by their start time
    df['case_start_time'] = df.groupby(case_id)[timestamp].transform('min')
    df = df.sort_values(['case_start_time', case_id, timestamp], ascending=True, kind='mergesort')
    df = df.drop(columns=['case_start_time'])
    return df.reset_index(drop=True)

def add_time_features(df, case_id, timestamp, activity):
    """Adds SuTraN-like time features."""
    start_time = time.time()
    print("Sorting log for time features...")
    # Ensure sorting before calculating shifts
    df = df.sort_values([case_id, timestamp], ascending=True, kind='mergesort').reset_index(drop=True)

    print("Computing case lengths...")
    # Case length (number of events in case)
    df['case_length'] = df.groupby(case_id, sort=False)[activity].transform('size')

    print("Computing time features...")
    # Time delta from previous event (ts_prev)
    df['prev_timestamp'] = df.groupby(case_id, sort=False)[timestamp].shift(1)
    df['ts_prev'] = (df[timestamp] - df['prev_timestamp']).dt.total_seconds()
    df['ts_prev'] = df['ts_prev'].fillna(0.0) # First event has 0 delta

    # Time delta until next event (tt_next)
    df['next_timestamp'] = df.groupby(case_id, sort=False)[timestamp].shift(-1)
    df['tt_next'] = (df['next_timestamp'] - df[timestamp]).dt.total_seconds()
    # Last event's tt_next will be NaN - keep it for now, will be 0 after END token add

    # Time from case start (ts_start)
    df['first_timestamp'] = df.groupby(case_id, sort=False)[timestamp].transform('min')
    df['ts_start'] = (df[timestamp] - df['first_timestamp']).dt.total_seconds()

    # Remaining time in case (rtime)
    df['last_timestamp'] = df.groupby(case_id, sort=False)[timestamp].transform('max')
    df['rtime'] = (df['last_timestamp'] - df[timestamp]).dt.total_seconds()

    # Ensure non-negative deltas (handle potential timestamp noise)
    for col in ['ts_prev', 'tt_next', 'rtime', 'ts_start']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Drop intermediate columns
    df = df.drop(columns=['prev_timestamp', 'next_timestamp', 'first_timestamp', 'last_timestamp'])

    # tt_next for last event should be 0. It's currently NaN.
    # This will be handled implicitly when the END token row is added later,
    # or when generating timeLabel_df, where NaN times are filled.
    # Let's fillna for ts_prev here.
    df['ts_prev'] = df['ts_prev'].fillna(0.0)

    print(f"Time features added in {time.time() - start_time:.2f} seconds")
    return df

def treat_categorical_features(train_df, test_df, val_df=None, categorical_columns=None):
    """
    Handles categorical features including activity labels.
    Maps categories to integers (starting from 1, 0 is reserved for padding).
    Handles missing values ('MISSINGVL') and OOV tokens ('<unk>').
    """
    start_time = time.time()
    print("Processing categorical features...")
    cardinality_dict = {}
    categorical_mapping_dict = {}

    if categorical_columns is None: categorical_columns = []
    if ACTIVITY_COL not in categorical_columns:
        categorical_columns.append(ACTIVITY_COL)

    all_dfs = [train_df, test_df]
    if val_df is not None: all_dfs.append(val_df)

    # Check for missing values ACROSS all datasets for each column
    missing_value_catcols = []
    for col in categorical_columns:
        has_missing = any(df[col].isna().any() for df in all_dfs)
        if has_missing:
            missing_value_catcols.append(col)
            print(f"  Column '{col}' has missing values. Adding 'MISSINGVL'.")
            for df in all_dfs:
                # Ensure column is object type before filling NA
                if pd.api.types.is_categorical_dtype(df[col]):
                    # Add 'MISSINGVL' category if it doesn't exist
                    if 'MISSINGVL' not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories('MISSINGVL')
                else:
                    # Convert to string first if not categorical
                    df[col] = df[col].astype(str)
                df[col].fillna('MISSINGVL', inplace=True)


    # --- Mapping ---
    for col in tqdm(categorical_columns, desc="Mapping Categoricals"):
        # Convert to string to handle mixed types before getting unique values
        train_df[col] = train_df[col].astype(str)
        unique_train_values = sorted(list(train_df[col].unique())) # Sort for consistent mapping

        cat_to_int = {}
        current_id = 1 # Start IDs from 1, 0 is for padding

        # Handle missing value token first if present
        if 'MISSINGVL' in unique_train_values:
            cat_to_int['MISSINGVL'] = current_id
            unique_train_values.remove('MISSINGVL')
            current_id += 1

        # Map remaining training values
        for value in unique_train_values:
            cat_to_int[value] = current_id
            current_id += 1

        # --- OOV Handling ---
        # Find values in test/val not in train (excluding MISSINGVL)
        oov_token = '<unk>' # Use a specific token for unseen values
        oov_id = current_id
        has_oov = False

        for df in [test_df, val_df] if val_df is not None else [test_df]:
             df[col] = df[col].astype(str) # Ensure string type
             unique_other = set(df[col].unique())
             oov_values = unique_other - set(cat_to_int.keys())
             if oov_values:
                 has_oov = True
                 print(f"  OOV values found in '{col}' in {'validation' if df is val_df else 'test'} set: {oov_values}")
                 for oov_val in oov_values:
                     cat_to_int[oov_val] = oov_id # Map all OOV to the same ID

        if has_oov:
            cat_to_int[oov_token] = oov_id # Add the OOV mapping itself
            current_id += 1

        # Apply mapping to all datasets
        for df in all_dfs:
            # Use map for efficiency, fill NA resulting from map with OOV id if needed
            df[col] = df[col].map(cat_to_int)
            if has_oov:
                 df[col].fillna(oov_id, inplace=True) # Fill any potential NAs from mapping OOV
            df[col] = df[col].astype(int) # Convert to integer

        # Store cardinality and mapping
        # Cardinality is the highest assigned ID + 1 (because of padding at 0)
        cardinality_dict[col] = current_id # This is the size excluding padding
        categorical_mapping_dict[col] = cat_to_int

    print(f"Categorical features processed in {time.time() - start_time:.2f} seconds")
    return train_df, test_df, val_df, cardinality_dict, categorical_mapping_dict


def create_validation_split(train_df, case_id, timestamp, val_split=0.2):
    """Creates validation split based on case start times."""
    start_time = time.time()
    print("Creating validation split...")
    case_start_df = train_df.groupby(case_id)[timestamp].min().reset_index()
    case_start_df = case_start_df.sort_values(by=timestamp, ascending=True)
    total_cases = len(case_start_df)
    val_cases = int(total_cases * val_split)
    train_cases = total_cases - val_cases
    train_case_ids = set(case_start_df[case_id].iloc[:train_cases].values)
    val_case_ids = set(case_start_df[case_id].iloc[train_cases:].values)

    # Efficient filtering
    train_filtered = train_df[train_df[case_id].isin(train_case_ids)].copy().reset_index(drop=True)
    val_filtered = train_df[train_df[case_id].isin(val_case_ids)].copy().reset_index(drop=True)

    print(f"Training set: {len(train_filtered)} events, {len(train_case_ids)} cases")
    print(f"Validation set: {len(val_filtered)} events, {len(val_case_ids)} cases")
    print(f"Validation split created in {time.time() - start_time:.2f} seconds")
    return train_filtered, val_filtered

def preprocess_numericals(train_df, val_df, test_df, numerical_cols, apply_log=False):
    """Standardizes numerical features."""
    start_time = time.time()
    print(f"Processing numerical features (Apply log: {apply_log})...")
    means_dict = {}
    std_dict = {}
    scalers = {}
    log_transformed_cols = []

    all_dfs = [train_df, val_df, test_df]

    # Identify log-transformable columns (non-negative across all splits)
    if apply_log:
        for col in numerical_cols:
            is_transformable = True
            for df in all_dfs:
                if not (df[col].isna() | (df[col] >= 0)).all():
                    is_transformable = False
                    break
            if is_transformable:
                log_transformed_cols.append(col)
        print(f"  Applying log1p to: {log_transformed_cols}")
        for col in log_transformed_cols:
            for df in all_dfs:
                df[col] = np.log1p(df[col])

    # Standardize
    for col in numerical_cols:
        scaler = StandardScaler()
        # Fit only on non-NaN values from the training set
        train_vals = train_df[col].dropna().values.reshape(-1, 1)
        if len(train_vals) > 0: # Ensure there's data to fit
             scaler.fit(train_vals)
             means_dict[col] = scaler.mean_[0]
             std_dict[col] = np.sqrt(scaler.var_[0]) if scaler.var_[0] > 1e-9 else 1.0 # Avoid division by zero std
             scalers[col] = scaler # Store scaler if needed later

             # Transform all datasets, handle potential NaNs during transform
             for df in all_dfs:
                 # Transform only non-NaN values and put them back
                 non_nan_mask = ~df[col].isna()
                 if non_nan_mask.any(): # Check if there are any non-NaN values to transform
                     vals_to_transform = df.loc[non_nan_mask, col].values.reshape(-1, 1)
                     transformed_vals = scaler.transform(vals_to_transform).flatten()
                     df.loc[non_nan_mask, col] = transformed_vals
        else:
            print(f"  Warning: Column '{col}' has no non-NaN values in training set. Skipping standardization.")
            means_dict[col] = 0.0
            std_dict[col] = 1.0

    # --- Missing Value Imputation (AFTER Standardization) ---
    # Add indicator columns and impute NAs with 0 (mean after standardization)
    indicator_cols_added = []
    print("Handling missing numerical values...")
    for col in numerical_cols:
        has_missing = any(df[col].isna().any() for df in all_dfs)
        if has_missing:
            indicator_col_name = f"{col}_missing"
            print(f"  Adding indicator column: {indicator_col_name}")
            indicator_cols_added.append(indicator_col_name)
            for df in all_dfs:
                df[indicator_col_name] = df[col].isna().astype(int)
                df[col].fillna(0.0, inplace=True) # Impute with standardized mean (0)

    print(f"Numerical features processed in {time.time() - start_time:.2f} seconds")
    return train_df, val_df, test_df, means_dict, std_dict, scalers, log_transformed_cols, indicator_cols_added


# --- Functions adapted from author's prefix_suffix_creation.py ---
# def create_prefix_suffixes_df(df, window_size, case_id, timestamp, activity_col,
#                                cat_casefts, num_casefts, cat_eventfts, num_eventfts, outcome):
#     """Creates intermediate DataFrames for prefixes, suffixes, and labels."""
#     start_time = time.time()
#     print("Creating intermediate prefix/suffix DataFrames...")

#     # Select relevant columns for each part
#     prefix_cols = [case_id, activity_col, 'ts_start', 'ts_prev', 'case_length'] + cat_casefts + num_casefts + cat_eventfts + num_eventfts
#     prefix_subset = df[prefix_cols].copy()

#     suffix_cols = [case_id, activity_col, 'ts_start', 'ts_prev', 'case_length']
#     suffix_subset = df[suffix_cols].copy()

#     actLabel_cols = [case_id, activity_col, 'case_length']
#     actLabel_subset = df[actLabel_cols].reset_index(drop = True).copy()

#     # Add END_TOKEN to activity labels
#     eos_token_int = len(np.unique(df[activity_col])) + 1 # Assign next integer ID
#     case_data = df.drop_duplicates(subset=case_id).reset_index(drop=True)[[case_id, 'case_length']]
#     end_df = pd.DataFrame({
#         case_id: case_data[case_id],
#         activity_col: eos_token_int, # Use integer representation directly
#         'case_length': case_data['case_length']
#     })
#     actLabel_subset = pd.concat([actLabel_subset, end_df], ignore_index=True)
#     # Add event index within case to sort correctly after concat
#     actLabel_subset['temp_evt_idx'] = actLabel_subset.groupby(case_id).cumcount()
#     actLabel_subset = actLabel_subset.sort_values(by=[case_id, 'temp_evt_idx']).drop(columns=['temp_evt_idx'])

#     timeLabel_cols = [case_id, 'case_length', 'tt_next', 'rtime']
#     timeLabel_subset = df[timeLabel_cols].copy()
#     # Add a row for the END token with 0 time values
#     end_time_df = pd.DataFrame({
#          case_id: case_data[case_id],
#          'case_length': case_data['case_length'],
#          'tt_next': 0.0,
#          'rtime': 0.0
#      })
#     timeLabel_subset = pd.concat([timeLabel_subset, end_time_df], ignore_index=True)
#     timeLabel_subset['temp_evt_idx'] = timeLabel_subset.groupby(case_id).cumcount()
#     timeLabel_subset = timeLabel_subset.sort_values(by=[case_id, 'temp_evt_idx']).drop(columns=['temp_evt_idx'])
#     timeLabel_subset.fillna({'tt_next': 0.0}, inplace=True) # Fill NaNs from original last events

#     if outcome:
#         outcomeLabel_cols = [case_id, 'case_length', outcome]
#         outcomeLabel_subset = df[outcomeLabel_cols].copy()

#     # --- Generate prefix-suffix pairs for each prefix length ---
#     prefix_df_list = []
#     suffix_df_list = []
#     timeLabel_df_list = []
#     actLabel_df_list = []
#     outcomeLabel_df_list = [] if outcome else None

#     # Get unique case IDs to iterate over efficiently
#     unique_cases = df[case_id].unique()
#     case_group_map = {case: group for case, group in df.groupby(case_id)}
#     actlabel_group_map = {case: group for case, group in actLabel_subset.groupby(case_id)}
#     timelabel_group_map = {case: group for case, group in timeLabel_subset.groupby(case_id)}
#     if outcome:
#         outcome_group_map = {case: group for case, group in outcomeLabel_subset.groupby(case_id)}

#     # Use tqdm for progress bar
#     for case, group in tqdm(case_group_map.items(), desc="Generating Prefix/Suffix Pairs"):
#         case_len = group['case_length'].iloc[0]
#         if case_len <= 1: continue # Skip single-event cases

#         act_labels_case = actlabel_group_map[case]
#         time_labels_case = timelabel_group_map[case]
#         if outcome:
#              outcome_label_case = outcome_group_map[case].iloc[0] # Outcome is case-level

#         max_pref_len = min(case_len, window_size) # Max prefix is the case len or window

#         for k in range(1, max_pref_len + 1): # Iterate through prefix lengths 1 to max_pref_len
#             prefix_id = f"{case}_{k}" # Unique ID for this prefix-suffix pair

#             # --- Prefix ---
#             prefix_k = group.head(k).copy()
#             prefix_k[case_id] = prefix_id # Assign new prefix ID
#             prefix_df_list.append(prefix_k)

#             # --- Suffix (Input for Decoder during training) ---
#             # Starts from the k-th event (last prefix event) as SOS token
#             suffix_k = group.iloc[k-1:].copy()
#             suffix_k[case_id] = prefix_id
#             suffix_df_list.append(suffix_k)

#             # --- Activity Labels (shifted suffix) ---
#             # Starts from k+1 event, includes EOS
#             act_label_k = act_labels_case.iloc[k:].copy()
#             act_label_k[case_id] = prefix_id
#             actLabel_df_list.append(act_label_k)

#             # --- Time Labels (shifted suffix) ---
#             # Starts from k+1 event, includes EOS times (0)
#             time_label_k = time_labels_case.iloc[k:].copy()
#             time_label_k[case_id] = prefix_id
#             timeLabel_df_list.append(time_label_k)

#             # --- Outcome Label ---
#             if outcome:
#                 outcome_label_k = outcome_label_case.copy()
#                 outcome_label_k[case_id] = prefix_id
#                 outcomeLabel_df_list.append(outcome_label_k)

#     # Concatenate results
#     prefix_df_final = pd.concat(prefix_df_list, ignore_index=True)
#     suffix_df_final = pd.concat(suffix_df_list, ignore_index=True)
#     timeLabel_df_final = pd.concat(timeLabel_df_list, ignore_index=True)
#     actLabel_df_final = pd.concat(actLabel_df_list, ignore_index=True)
#     outcomeLabel_df_final = pd.concat(outcomeLabel_df_list, ignore_index=True) if outcome else None

#     print(f"Intermediate DFs created in {time.time() - start_time:.2f} seconds")

#     if outcome:
#         return prefix_df_final, suffix_df_final, timeLabel_df_final, actLabel_df_final, outcomeLabel_df_final
#     else:
#         return prefix_df_final, suffix_df_final, timeLabel_df_final, actLabel_df_final
def create_prefix_suffixes_df(df, window_size, case_id, timestamp, activity_col,
                               cat_casefts, num_casefts, cat_eventfts, num_eventfts, outcome):
    """Creates intermediate DataFrames for prefixes, suffixes, and labels."""
    start_time = time.time()
    print("Creating intermediate prefix/suffix DataFrames...")

    # --- Select relevant columns (Make sure 'ts_start' is available if needed later) ---
    # Note: 'ts_start' is required for sorting in generate_*_tensors functions
    # Ensure 'ts_start' is created in add_time_features and included here
    prefix_cols = [case_id, activity_col, 'ts_start', 'ts_prev', 'case_length'] + cat_casefts + num_casefts + cat_eventfts + num_eventfts
    # Check if all prefix_cols exist
    missing_prefix_cols = [col for col in prefix_cols if col not in df.columns]
    if missing_prefix_cols:
        print(f"Warning: Missing columns in input df for prefix_subset: {missing_prefix_cols}")
        prefix_cols = [col for col in prefix_cols if col in df.columns] # Use only existing cols
    prefix_subset = df[prefix_cols].copy()

    suffix_cols = [case_id, activity_col, 'ts_start', 'ts_prev', 'case_length']
    missing_suffix_cols = [col for col in suffix_cols if col not in df.columns]
    if missing_suffix_cols:
        print(f"Warning: Missing columns in input df for suffix_subset: {missing_suffix_cols}")
        suffix_cols = [col for col in suffix_cols if col in df.columns] # Use only existing cols
    suffix_subset = df[suffix_cols].copy()

    actLabel_cols = [case_id, activity_col, 'case_length', 'ts_start'] # Keep ts_start for sorting consistency
    missing_act_cols = [col for col in actLabel_cols if col not in df.columns]
    if missing_act_cols:
        print(f"Warning: Missing columns in input df for actLabel_subset: {missing_act_cols}")
        actLabel_cols = [col for col in actLabel_cols if col in df.columns]
    actLabel_subset = df[actLabel_cols].reset_index(drop = True).copy()

    # --- Add END_TOKEN to activity labels ---
    # Determine EOS token integer ID based on MAX existing activity ID + 1
    # Ensure activity_col is integer type before finding max
    if not pd.api.types.is_numeric_dtype(df[activity_col]):
         print(f"Warning: Activity column '{activity_col}' is not numeric. Cannot reliably determine EOS token ID. Assuming max+1 logic holds.")
         # Fallback or raise error depending on requirements
         eos_token_int = df[activity_col].astype('category').cat.codes.max() + 1 + 1 # +1 for mapping start, +1 for EOS
    else:
        eos_token_int = df[activity_col].max() + 1

    case_data = df.drop_duplicates(subset=case_id).reset_index(drop=True)[[case_id, 'case_length', 'ts_start']] # Keep ts_start
    # Ensure ts_start for END token makes sense (e.g., last event's ts_start + tiny epsilon or NaN/0?)
    # Using last event's ts_start for simplicity, might need adjustment based on meaning
    last_ts_start = df.loc[df.groupby(case_id)['ts_start'].idxmax()][[case_id, 'ts_start']].set_index(case_id)
    case_data = case_data.join(last_ts_start, on=case_id, rsuffix='_last')

    end_df = pd.DataFrame({
        case_id: case_data[case_id],
        activity_col: eos_token_int,
        'case_length': case_data['case_length'],
        'ts_start': case_data['ts_start_last'] # Use last ts_start for sorting end token correctly
    })
    actLabel_subset = pd.concat([actLabel_subset, end_df], ignore_index=True)
    # Use cumcount for reliable ordering within case after concat
    actLabel_subset['temp_evt_idx'] = actLabel_subset.groupby(case_id).cumcount()
    # Sort by case and then by the event order (ts_start or temp_evt_idx)
    actLabel_subset = actLabel_subset.sort_values(by=[case_id, 'ts_start', 'temp_evt_idx']).drop(columns=['temp_evt_idx']) # Added ts_start


    # --- Time Labels ---
    timeLabel_cols = [case_id, 'case_length', 'tt_next', 'rtime', 'ts_start'] # Keep ts_start
    missing_time_cols = [col for col in timeLabel_cols if col not in df.columns]
    if missing_time_cols:
        print(f"Warning: Missing columns in input df for timeLabel_subset: {missing_time_cols}")
        timeLabel_cols = [col for col in timeLabel_cols if col in df.columns]
    timeLabel_subset = df[timeLabel_cols].copy()

    # Add a row for the END token with 0 time values
    end_time_df = pd.DataFrame({
         case_id: case_data[case_id],
         'case_length': case_data['case_length'],
         'tt_next': 0.0,
         'rtime': 0.0,
         'ts_start': case_data['ts_start_last'] # Use same ts_start as activity END token
     })
    timeLabel_subset = pd.concat([timeLabel_subset, end_time_df], ignore_index=True)
    # Use cumcount for reliable ordering
    timeLabel_subset['temp_evt_idx'] = timeLabel_subset.groupby(case_id).cumcount()
    # Sort by case and then by the event order
    timeLabel_subset = timeLabel_subset.sort_values(by=[case_id, 'ts_start', 'temp_evt_idx']).drop(columns=['temp_evt_idx']) # Added ts_start
    timeLabel_subset.fillna({'tt_next': 0.0}, inplace=True) # Fill NaNs from original last events


    if outcome:
        # Ensure ts_start is present if needed later
        outcomeLabel_cols = [case_id, 'case_length', outcome, 'ts_start']
        missing_outcome_cols = [col for col in outcomeLabel_cols if col not in df.columns]
        if missing_outcome_cols:
             print(f"Warning: Missing columns in input df for outcomeLabel_subset: {missing_outcome_cols}")
             outcomeLabel_cols = [col for col in outcomeLabel_cols if col in df.columns]
        outcomeLabel_subset = df[outcomeLabel_cols].copy()

    # --- Generate prefix-suffix pairs for each prefix length ---
    prefix_df_list = []
    suffix_df_list = []
    timeLabel_df_list = []
    actLabel_df_list = []
    outcomeLabel_df_list = [] if outcome else None

    # Use original df for grouping to avoid issues with modified subsets
    grouped_orig = df.groupby(case_id)
    # Pre-group label subsets for faster lookup
    actlabel_grouped = actLabel_subset.groupby(case_id)
    timelabel_grouped = timeLabel_subset.groupby(case_id)
    if outcome:
        # Outcome is case level, get it once per case
        outcome_map = df.drop_duplicates(subset=case_id).set_index(case_id)[outcome]


    for case, group in tqdm(grouped_orig, desc="Generating Prefix/Suffix Pairs"):
        case_len = group['case_length'].iloc[0]
        if case_len <= 1: continue

        # Use the pre-created subsets for labels
        act_labels_case = actlabel_grouped.get_group(case)
        time_labels_case = timelabel_grouped.get_group(case)
        if outcome:
            outcome_label_val = outcome_map.get(case, None) # Get single outcome value

        # Re-select prefix and suffix subsets based on original group to ensure all columns
        current_prefix_subset = group[prefix_subset.columns].copy()
        current_suffix_subset = group[suffix_subset.columns].copy()


        max_pref_len = min(case_len, window_size)

        for k in range(1, max_pref_len + 1):
            prefix_id = f"{case}_{k}"

            # --- Prefix ---
            prefix_k = current_prefix_subset.head(k).copy()
            prefix_k[case_id] = prefix_id
            prefix_df_list.append(prefix_k)

            # --- Suffix (Input for Decoder during training) ---
            suffix_k = current_suffix_subset.iloc[k-1:].copy() # Starts from k-th event
            suffix_k[case_id] = prefix_id
            suffix_df_list.append(suffix_k)

            # --- Activity Labels (shifted suffix) ---
            act_label_k = act_labels_case.iloc[k:].copy() # Starts from k+1 event (incl. EOS)
            act_label_k[case_id] = prefix_id
            actLabel_df_list.append(act_label_k)

            # --- Time Labels (shifted suffix) ---
            time_label_k = time_labels_case.iloc[k:].copy() # Starts from k+1 event (incl. EOS times)
            time_label_k[case_id] = prefix_id
            timeLabel_df_list.append(time_label_k)

            # --- Outcome Label ---
            if outcome:
                # Create a small DataFrame for the single outcome value for this prefix_id
                outcome_label_df_k = pd.DataFrame([{case_id: prefix_id, outcome: outcome_label_val}])
                outcomeLabel_df_list.append(outcome_label_df_k)


    # Concatenate results
    prefix_df_final = pd.concat(prefix_df_list, ignore_index=True) if prefix_df_list else pd.DataFrame(columns=prefix_subset.columns)
    suffix_df_final = pd.concat(suffix_df_list, ignore_index=True) if suffix_df_list else pd.DataFrame(columns=suffix_subset.columns)
    timeLabel_df_final = pd.concat(timeLabel_df_list, ignore_index=True) if timeLabel_df_list else pd.DataFrame(columns=timeLabel_subset.columns)
    actLabel_df_final = pd.concat(actLabel_df_list, ignore_index=True) if actLabel_df_list else pd.DataFrame(columns=actLabel_subset.columns)
    outcomeLabel_df_final = pd.concat(outcomeLabel_df_list, ignore_index=True) if outcome and outcomeLabel_df_list else None

    print(f"Intermediate DFs created in {time.time() - start_time:.2f} seconds")

    # Ensure 'ts_start' exists before returning (important for sorting in next step)
    if 'ts_start' not in timeLabel_df_final.columns:
         print("Error: 'ts_start' column missing from final timeLabel_df!")
         # Potentially re-add it if needed, though sorting might use evt_idx now
    if 'ts_start' not in actLabel_df_final.columns:
         print("Error: 'ts_start' column missing from final actLabel_df!")


    if outcome:
        return prefix_df_final, suffix_df_final, timeLabel_df_final, actLabel_df_final, outcomeLabel_df_final
    else:
        return prefix_df_final, suffix_df_final, timeLabel_df_final, actLabel_df_final

# --- Functions adapted from author's tensor_creation.py ---
def generate_prefix_tensors(prefix_df, cat_cols, num_cols, window_size, case_id_col, activity_col):
    """Creates prefix tensors from the prefix DataFrame."""
    start_time = time.time()
    print("  Generating prefix tensors...")

    # Ensure sorting for correct indexing
    prefix_df = prefix_df.sort_values(by=[case_id_col, 'ts_start']).reset_index(drop=True)

    # Get unique prefix IDs and map them to integers
    prefix_ids = prefix_df[case_id_col].unique()
    num_prefs = len(prefix_ids)
    str_to_int = {pid: i for i, pid in enumerate(prefix_ids)}
    prefix_df['prefix_id_int'] = prefix_df[case_id_col].map(str_to_int)

    # Calculate event index within each prefix
    prefix_df['evt_idx'] = prefix_df.groupby('prefix_id_int', sort=False).cumcount()

    # Create indices tensor for scatter updates
    idx = torch.from_numpy(prefix_df[['prefix_id_int', 'evt_idx']].values)
    if idx.max() >= num_prefs or idx.min() < 0:
         print("Index out of bounds detected in prefix tensor creation!")
         # Handle error or investigate prefix_id_int/evt_idx calculation

    # Create tensors
    prefix_tensors = []
    # Categorical Tensors (ensure activity is handled correctly if in cat_cols)
    for col in cat_cols:
        cat_tens = torch.zeros((num_prefs, window_size), dtype=torch.long)
        updates = torch.from_numpy(prefix_df[col].values).long()
        # Clamp indices to prevent out-of-bounds errors during scatter
        safe_idx_0 = torch.clamp(idx[:, 0], 0, num_prefs - 1)
        safe_idx_1 = torch.clamp(idx[:, 1], 0, window_size - 1)
        cat_tens[safe_idx_0, safe_idx_1] = updates
        prefix_tensors.append(cat_tens)

    # Numerical Tensor
    if num_cols:
        num_nums = len(num_cols)
        num_tens = torch.zeros((num_prefs, window_size, num_nums), dtype=torch.float)
        updates = torch.from_numpy(prefix_df[num_cols].values).float()
        safe_idx_0 = torch.clamp(idx[:, 0], 0, num_prefs - 1)
        safe_idx_1 = torch.clamp(idx[:, 1], 0, window_size - 1)
        num_tens[safe_idx_0, safe_idx_1] = updates
        prefix_tensors.append(num_tens)
    else:
         # Append an empty tensor if no numerical features
         prefix_tensors.append(torch.zeros((num_prefs, window_size, 0), dtype=torch.float))


    # Padding Mask (True where padded)
    padding_mask = torch.ones((num_prefs, window_size), dtype=torch.bool)
    safe_idx_0 = torch.clamp(idx[:, 0], 0, num_prefs - 1)
    safe_idx_1 = torch.clamp(idx[:, 1], 0, window_size - 1)
    padding_mask[safe_idx_0, safe_idx_1] = False
    prefix_tensors.append(padding_mask)

    print(f"  Prefix tensors generated in {time.time() - start_time:.2f}s")
    return tuple(prefix_tensors), str_to_int # Return mapping for consistency

def generate_suffix_tensors(suffix_df, cat_cols, num_cols, window_size, case_id_col, activity_col, str_to_int_map):
    """Creates suffix input tensors (for decoder)."""
    start_time = time.time()
    print("  Generating suffix tensors...")
    suffix_df = suffix_df.sort_values(by=[case_id_col, 'ts_start']).reset_index(drop=True)
    num_prefs = len(str_to_int_map)
    suffix_df['prefix_id_int'] = suffix_df[case_id_col].map(str_to_int_map)
    suffix_df['evt_idx'] = suffix_df.groupby('prefix_id_int', sort=False).cumcount()
    idx = torch.from_numpy(suffix_df[['prefix_id_int', 'evt_idx']].values)

    suffix_tensors = []
    # Categorical
    for col in cat_cols:
        cat_tens = torch.zeros((num_prefs, window_size), dtype=torch.long)
        updates = torch.from_numpy(suffix_df[col].values).long()
        safe_idx_0 = torch.clamp(idx[:, 0], 0, num_prefs - 1)
        safe_idx_1 = torch.clamp(idx[:, 1], 0, window_size - 1)
        cat_tens[safe_idx_0, safe_idx_1] = updates
        suffix_tensors.append(cat_tens)

    # Numerical
    if num_cols:
        num_nums = len(num_cols)
        # Use -1 as padding indicator, consistent with author? Check tensor_creation.py
        # Author uses 0 for prefix/suffix numerics, -100 for time labels. Let's use 0.
        num_tens = torch.zeros((num_prefs, window_size, num_nums), dtype=torch.float)
        updates = torch.from_numpy(suffix_df[num_cols].values).float()
        safe_idx_0 = torch.clamp(idx[:, 0], 0, num_prefs - 1)
        safe_idx_1 = torch.clamp(idx[:, 1], 0, window_size - 1)
        num_tens[safe_idx_0, safe_idx_1] = updates
        suffix_tensors.append(num_tens)
    else:
        suffix_tensors.append(torch.zeros((num_prefs, window_size, 0), dtype=torch.float))

    print(f"  Suffix tensors generated in {time.time() - start_time:.2f}s")
    return tuple(suffix_tensors)

# def generate_label_tensors(time_label_df, act_label_df, num_cols_time, act_col, window_size, case_id_col, str_to_int_map, outcome_label_df=None, outcome_col=None):
#     """Creates label tensors (time, activity, optional outcome)."""
#     start_time = time.time()
#     print("  Generating label tensors...")
#     num_prefs = len(str_to_int_map)

#     # Time Labels
#     time_label_df = time_label_df.sort_values(by=[case_id_col, 'ts_start']).reset_index(drop=True)
#     time_label_df['prefix_id_int'] = time_label_df[case_id_col].map(str_to_int_map)
#     time_label_df['evt_idx'] = time_label_df.groupby('prefix_id_int', sort=False).cumcount()
#     idx_time = torch.from_numpy(time_label_df[['prefix_id_int', 'evt_idx']].values)
#     num_time_labels = len(num_cols_time)
#     time_label_tens = torch.full((num_prefs, window_size, num_time_labels), -100.0, dtype=torch.float) # Use -100 padding
#     updates_time = torch.from_numpy(time_label_df[num_cols_time].values).float()
#     safe_idx_0_time = torch.clamp(idx_time[:, 0], 0, num_prefs - 1)
#     safe_idx_1_time = torch.clamp(idx_time[:, 1], 0, window_size - 1)
#     time_label_tens[safe_idx_0_time, safe_idx_1_time] = updates_time
#     # Split into tt_next and rtime tensors
#     ttnext_tens = time_label_tens[:, :, 0:1] # Keep dimension
#     rtime_tens = time_label_tens[:, :, 1:2]  # Keep dimension

#     # Activity Labels
#     act_label_df = act_label_df.sort_values(by=[case_id_col, 'ts_start']).reset_index(drop=True)
#     act_label_df['prefix_id_int'] = act_label_df[case_id_col].map(str_to_int_map)
#     act_label_df['evt_idx'] = act_label_df.groupby('prefix_id_int', sort=False).cumcount()
#     idx_act = torch.from_numpy(act_label_df[['prefix_id_int', 'evt_idx']].values)
#     act_label_tens = torch.zeros((num_prefs, window_size), dtype=torch.long) # 0 padding
#     updates_act = torch.from_numpy(act_label_df[act_col].values).long()
#     safe_idx_0_act = torch.clamp(idx_act[:, 0], 0, num_prefs - 1)
#     safe_idx_1_act = torch.clamp(idx_act[:, 1], 0, window_size - 1)
#     act_label_tens[safe_idx_0_act, safe_idx_1_act] = updates_act

#     label_tensors = (ttnext_tens, rtime_tens, act_label_tens)

#     # Outcome Label (if applicable)
#     if outcome_label_df is not None and outcome_col:
#         # Outcome is case-level, so take the value for each prefix_id_int once
#         outcome_label_df = outcome_label_df.sort_values(by=[case_id_col]).reset_index(drop=True)
#         outcome_label_df['prefix_id_int'] = outcome_label_df[case_id_col].map(str_to_int_map)
#         # Ensure one value per prefix_id_int - drop duplicates based on the int ID
#         outcome_map_df = outcome_label_df.drop_duplicates(subset=['prefix_id_int'])
#         outcome_label_tens_ordered = torch.zeros(num_prefs, 1, dtype=torch.float)
#         # Map values based on the integer index
#         outcome_vals = torch.from_numpy(outcome_map_df[outcome_col].values).float().unsqueeze(1)
#         indices_out = torch.from_numpy(outcome_map_df['prefix_id_int'].values).long()
#         # Ensure indices are within bounds before scattering
#         safe_indices_out = torch.clamp(indices_out, 0, num_prefs - 1)
#         outcome_label_tens_ordered[safe_indices_out] = outcome_vals[torch.arange(len(safe_indices_out))] # Usearange for correct assignment

#         label_tensors += (outcome_label_tens_ordered,)

#     print(f"  Label tensors generated in {time.time() - start_time:.2f}s")
#     return label_tensors
def generate_label_tensors(time_label_df, act_label_df, num_cols_time, act_col, window_size, case_id_col, str_to_int_map, outcome_label_df=None, outcome_col=None):
    """Creates label tensors (time, activity, optional outcome)."""
    start_time = time.time()
    print("  Generating label tensors...")
    num_prefs = len(str_to_int_map)

    # Time Labels
    # Ensure 'ts_start' exists before attempting to sort
    if 'ts_start' not in time_label_df.columns:
        print("Error in generate_label_tensors: 'ts_start' column missing in time_label_df. Cannot sort.")
        # Decide how to proceed: raise error, skip sort, or use another column
        # Option: Skip sorting if 'ts_start' is missing
        time_label_df = time_label_df.sort_values(by=[case_id_col]).reset_index(drop=True) # Sort only by case_id
    else:
        # Sort by case_id and ts_start if ts_start is present
        time_label_df = time_label_df.sort_values(by=[case_id_col, 'ts_start']).reset_index(drop=True)

    time_label_df['prefix_id_int'] = time_label_df[case_id_col].map(str_to_int_map)
    # Check for NaNs introduced by map (if some IDs weren't in the map)
    if time_label_df['prefix_id_int'].isna().any():
        print(f"Warning: NaNs found in prefix_id_int for time_label_df. Cases might be missing from str_to_int_map.")
        time_label_df.dropna(subset=['prefix_id_int'], inplace=True) # Drop rows that couldn't be mapped
        time_label_df['prefix_id_int'] = time_label_df['prefix_id_int'].astype(int) # Convert to int after dropna

    time_label_df['evt_idx'] = time_label_df.groupby('prefix_id_int', sort=False).cumcount()
    # Filter out rows where evt_idx >= window_size as they won't fit in the tensor
    time_label_df = time_label_df[time_label_df['evt_idx'] < window_size]
    idx_time = torch.from_numpy(time_label_df[['prefix_id_int', 'evt_idx']].values)

    num_time_labels = len(num_cols_time)
    time_label_tens = torch.full((num_prefs, window_size, num_time_labels), -100.0, dtype=torch.float) # Use -100 padding
    updates_time = torch.from_numpy(time_label_df[num_cols_time].values).float()
    # Ensure indices are within bounds AFTER filtering evt_idx
    safe_idx_0_time = torch.clamp(idx_time[:, 0], 0, num_prefs - 1)
    # safe_idx_1_time should be okay now due to the evt_idx < window_size filter
    safe_idx_1_time = idx_time[:, 1]
    time_label_tens[safe_idx_0_time, safe_idx_1_time] = updates_time
    # Split into tt_next and rtime tensors
    ttnext_tens = time_label_tens[:, :, 0:1] # Keep dimension
    rtime_tens = time_label_tens[:, :, 1:2]  # Keep dimension

    # Activity Labels
    if 'ts_start' not in act_label_df.columns:
        print("Error in generate_label_tensors: 'ts_start' column missing in act_label_df. Cannot sort.")
        act_label_df = act_label_df.sort_values(by=[case_id_col]).reset_index(drop=True)
    else:
        act_label_df = act_label_df.sort_values(by=[case_id_col, 'ts_start']).reset_index(drop=True)

    act_label_df['prefix_id_int'] = act_label_df[case_id_col].map(str_to_int_map)
    if act_label_df['prefix_id_int'].isna().any():
        print(f"Warning: NaNs found in prefix_id_int for act_label_df. Cases might be missing from str_to_int_map.")
        act_label_df.dropna(subset=['prefix_id_int'], inplace=True)
        act_label_df['prefix_id_int'] = act_label_df['prefix_id_int'].astype(int)


    act_label_df['evt_idx'] = act_label_df.groupby('prefix_id_int', sort=False).cumcount()
    # Filter evt_idx
    act_label_df = act_label_df[act_label_df['evt_idx'] < window_size]
    idx_act = torch.from_numpy(act_label_df[['prefix_id_int', 'evt_idx']].values)

    act_label_tens = torch.zeros((num_prefs, window_size), dtype=torch.long) # 0 padding
    updates_act = torch.from_numpy(act_label_df[act_col].values).long()
    safe_idx_0_act = torch.clamp(idx_act[:, 0], 0, num_prefs - 1)
    safe_idx_1_act = idx_act[:, 1]
    act_label_tens[safe_idx_0_act, safe_idx_1_act] = updates_act

    label_tensors = (ttnext_tens, rtime_tens, act_label_tens)

    # --- Outcome Label (No change needed here if structure is correct) ---
    if outcome_label_df is not None and outcome_col:
        # Outcome is case-level, so take the value for each prefix_id_int once
        outcome_label_df = outcome_label_df.sort_values(by=[case_id_col]).reset_index(drop=True)
        outcome_label_df['prefix_id_int'] = outcome_label_df[case_id_col].map(str_to_int_map)
        if outcome_label_df['prefix_id_int'].isna().any():
            print(f"Warning: NaNs found in prefix_id_int for outcome_label_df.")
            outcome_label_df.dropna(subset=['prefix_id_int'], inplace=True)
            outcome_label_df['prefix_id_int'] = outcome_label_df['prefix_id_int'].astype(int)

        # Ensure one value per prefix_id_int - drop duplicates based on the int ID
        outcome_map_df = outcome_label_df.drop_duplicates(subset=['prefix_id_int'])
        outcome_label_tens_ordered = torch.zeros(num_prefs, 1, dtype=torch.float)
        # Map values based on the integer index
        outcome_vals = torch.from_numpy(outcome_map_df[outcome_col].values).float().unsqueeze(1)
        indices_out = torch.from_numpy(outcome_map_df['prefix_id_int'].values).long()
        # Ensure indices are within bounds before scattering
        safe_indices_out = torch.clamp(indices_out, 0, num_prefs - 1)
        # Need to handle cases where indices might not cover all 0 to num_prefs-1
        valid_map_indices = torch.arange(len(safe_indices_out))
        outcome_label_tens_ordered[safe_indices_out] = outcome_vals[valid_map_indices]


        label_tensors += (outcome_label_tens_ordered,)


    print(f"  Label tensors generated in {time.time() - start_time:.2f}s")
    return label_tensors

# --- Main Processing Function (Modified) ---
def run_preprocessing_sutran_like(train_csv_path, test_csv_path, tensor_output_path, metadata_output_path,
                                 validation_split=0.2, window_size=None, apply_log_transform=False):
    """
    Processes raw data using author's logic principles and saves tensors.
    """
    total_start_time = time.time()
    print("--- Starting Modified Data Preprocessing ---")

    # --- Load Data ---
    try:
        df_train_raw = pd.read_csv(train_csv_path)
        df_test_raw = pd.read_csv(test_csv_path)
        print(f"Loaded data from {train_csv_path} and {test_csv_path}")
    except FileNotFoundError as e:
        print(f"Error: Cannot find data file {e.filename}. Exiting.")
        exit(1)

    # Convert timestamp column right after loading
    df_train_raw[TIMESTAMP_COL] = pd.to_datetime(df_train_raw[TIMESTAMP_COL])
    df_test_raw[TIMESTAMP_COL] = pd.to_datetime(df_test_raw[TIMESTAMP_COL])

    # --- Create output directory ---
    os.makedirs(os.path.dirname(tensor_output_path), exist_ok=True)

    # --- Add Time Features ---
    df_train = add_time_features(df_train_raw, CASE_ID_COL, TIMESTAMP_COL, ACTIVITY_COL)
    df_test = add_time_features(df_test_raw, CASE_ID_COL, TIMESTAMP_COL, ACTIVITY_COL)

    # --- Create validation split ---
    df_train, df_val = create_validation_split(df_train, CASE_ID_COL, TIMESTAMP_COL, validation_split)

    # --- Filter by case length (optional but recommended) ---
    win_size = window_size or WINDOW_SIZE
    if win_size:
        print(f"Filtering cases by maximum length: {win_size}")
        # Calculate length if not already present
        if 'case_length' not in df_train.columns:
             df_train['case_length'] = df_train.groupby(CASE_ID_COL, sort=False)[ACTIVITY_COL].transform('size')
        if 'case_length' not in df_val.columns:
             df_val['case_length'] = df_val.groupby(CASE_ID_COL, sort=False)[ACTIVITY_COL].transform('size')
        if 'case_length' not in df_test.columns:
             df_test['case_length'] = df_test.groupby(CASE_ID_COL, sort=False)[ACTIVITY_COL].transform('size')

        train_orig_cases = df_train[CASE_ID_COL].nunique()
        val_orig_cases = df_val[CASE_ID_COL].nunique()
        test_orig_cases = df_test[CASE_ID_COL].nunique()

        df_train = df_train[df_train['case_length'] <= win_size].reset_index(drop=True)
        df_val = df_val[df_val['case_length'] <= win_size].reset_index(drop=True)
        df_test = df_test[df_test['case_length'] <= win_size].reset_index(drop=True)

        print(f"  Train: {train_orig_cases} -> {df_train[CASE_ID_COL].nunique()} cases")
        print(f"  Val:   {val_orig_cases} -> {df_val[CASE_ID_COL].nunique()} cases")
        print(f"  Test:  {test_orig_cases} -> {df_test[CASE_ID_COL].nunique()} cases")


    # --- Define Feature Sets ---
    all_cat_features_base = CAT_CASE_FTS + CAT_EVENT_FTS
    basic_time_features = ['ts_prev', 'ts_start']
    all_num_features_base = NUM_CASE_FTS + NUM_EVENT_FTS
    time_label_features = ['tt_next', 'rtime'] # Order matters for tensor creation
    outcome_feature = None # Define if you have an outcome column

    # --- Process Categorical Features ---
    # Pass only base features, activity is handled separately in prefix/suffix creation
    df_train, df_test, df_val, cardinality_dict, cat_mapping_dict = treat_categorical_features(
        df_train, df_test, df_val, all_cat_features_base + [ACTIVITY_COL] # Include activity here
    )
    # Store activity cardinality separately for convenience
    activity_cardinality = cardinality_dict.get(ACTIVITY_COL, 0)
    eos_token_int = activity_cardinality # Next available ID (since mapping starts from 1)
    num_activities_with_special = activity_cardinality + 1 # +1 for padding

    # --- Process Numerical Features ---
    # Important: Standardize ONLY features present, not targets yet
    numerical_cols_to_process = basic_time_features + all_num_features_base
    df_train, df_val, df_test, means_dict, std_dict, scalers, log_transformed_cols, indicator_cols_added = preprocess_numericals(
        df_train, df_val, df_test, numerical_cols_to_process, apply_log=apply_log_transform
    )
    # Update the list of all numerical features to include indicators
    all_num_features_final = all_num_features_base + basic_time_features + indicator_cols_added

    # --- Standardize Time Labels Separately (as per author's approach) ---
    print("Standardizing time labels...")
    label_means = {}
    label_stds = {}
    for col in time_label_features:
         scaler_label = StandardScaler()
         train_vals_label = df_train[col].dropna().values.reshape(-1, 1)
         if len(train_vals_label) > 0:
             scaler_label.fit(train_vals_label)
             label_means[col] = scaler_label.mean_[0]
             label_stds[col] = np.sqrt(scaler_label.var_[0]) if scaler_label.var_[0] > 1e-9 else 1.0
             # Transform labels in all datasets
             for df in [df_train, df_val, df_test]:
                  non_nan_mask = ~df[col].isna()
                  if non_nan_mask.any():
                      vals_to_transform = df.loc[non_nan_mask, col].values.reshape(-1, 1)
                      df.loc[non_nan_mask, col] = scaler_label.transform(vals_to_transform).flatten()
                  df[col].fillna(-100.0, inplace=True) # Fill remaining NaNs with padding value AFTER scaling
         else:
             print(f"  Warning: Label '{col}' has no non-NaN values in training set. Using 0 mean, 1 std.")
             label_means[col] = 0.0
             label_stds[col] = 1.0
             # Fill all with padding value
             for df in [df_train, df_val, df_test]:
                 df[col].fillna(-100.0, inplace=True)


    # --- Create Intermediate DataFrames (Author's Method) ---
    # Define columns needed for the intermediate DFs based on processed features
    prefix_cat_cols = all_cat_features_base + [ACTIVITY_COL] # All cats used in prefix
    prefix_num_cols = all_num_features_final # All nums used in prefix
    suffix_cat_cols = [ACTIVITY_COL] # Only activity used in suffix input
    suffix_num_cols = basic_time_features # Only basic time used in suffix input

    # Generate intermediate DFs for train, val, test
    print("\n--- Generating Intermediate DataFrames ---")
    train_pref_df, train_suff_df, train_timeL_df, train_actL_df = create_prefix_suffixes_df(
        df_train, win_size, CASE_ID_COL, TIMESTAMP_COL, ACTIVITY_COL,
        CAT_CASE_FTS, NUM_CASE_FTS, CAT_EVENT_FTS, NUM_EVENT_FTS, outcome=None # Pass actual lists
    )
    val_pref_df, val_suff_df, val_timeL_df, val_actL_df = create_prefix_suffixes_df(
        df_val, win_size, CASE_ID_COL, TIMESTAMP_COL, ACTIVITY_COL,
        CAT_CASE_FTS, NUM_CASE_FTS, CAT_EVENT_FTS, NUM_EVENT_FTS, outcome=None
    )
    test_pref_df, test_suff_df, test_timeL_df, test_actL_df = create_prefix_suffixes_df(
        df_test, win_size, CASE_ID_COL, TIMESTAMP_COL, ACTIVITY_COL,
        CAT_CASE_FTS, NUM_CASE_FTS, CAT_EVENT_FTS, NUM_EVENT_FTS, outcome=None
    )

    # --- Convert DataFrames to Tensors (Author's Method) ---
    print("\n--- Generating Final Tensors ---")
    print("Processing Training Data...")
    train_prefix_tensors, str_to_int_train = generate_prefix_tensors(
        train_pref_df, prefix_cat_cols, prefix_num_cols, win_size, CASE_ID_COL, ACTIVITY_COL)
    train_suffix_tensors = generate_suffix_tensors(
        train_suff_df, suffix_cat_cols, suffix_num_cols, win_size, CASE_ID_COL, ACTIVITY_COL, str_to_int_train)
    train_label_tensors = generate_label_tensors(
        train_timeL_df, train_actL_df, time_label_features, ACTIVITY_COL, win_size, CASE_ID_COL, str_to_int_train)
    train_data_final = train_prefix_tensors + train_suffix_tensors + train_label_tensors

    print("Processing Validation Data...")
    val_prefix_tensors, str_to_int_val = generate_prefix_tensors(
        val_pref_df, prefix_cat_cols, prefix_num_cols, win_size, CASE_ID_COL, ACTIVITY_COL)
    val_suffix_tensors = generate_suffix_tensors(
        val_suff_df, suffix_cat_cols, suffix_num_cols, win_size, CASE_ID_COL, ACTIVITY_COL, str_to_int_val)
    val_label_tensors = generate_label_tensors(
        val_timeL_df, val_actL_df, time_label_features, ACTIVITY_COL, win_size, CASE_ID_COL, str_to_int_val)
    val_data_final = val_prefix_tensors + val_suffix_tensors + val_label_tensors

    print("Processing Test Data...")
    test_prefix_tensors, str_to_int_test = generate_prefix_tensors(
        test_pref_df, prefix_cat_cols, prefix_num_cols, win_size, CASE_ID_COL, ACTIVITY_COL)
    test_suffix_tensors = generate_suffix_tensors(
        test_suff_df, suffix_cat_cols, suffix_num_cols, win_size, CASE_ID_COL, ACTIVITY_COL, str_to_int_test)
    test_label_tensors = generate_label_tensors(
        test_timeL_df, test_actL_df, time_label_features, ACTIVITY_COL, win_size, CASE_ID_COL, str_to_int_test)
    test_data_final = test_prefix_tensors + test_suffix_tensors + test_label_tensors

    # --- Save Processed Tensors and Metadata ---
    print(f"\nSaving final tensors to {tensor_output_path}...")
    # Save as a dictionary of tuples, as expected by later scripts
    tensor_payload_final = {
        'train': train_data_final,
        'val': val_data_final,
        'test': test_data_final
    }
    torch.save(tensor_payload_final, tensor_output_path)
    print("Final tensors saved.")

    print(f"Saving metadata to {metadata_output_path}...")
    # Combine all metadata
    final_metadata = {
        'cardinality_dict': cardinality_dict, # Original cardinalities before padding + OOV
        'categorical_mapping_dict': cat_mapping_dict,
        'train_means': {**means_dict, **label_means}, # Combine feature and label means
        'train_stds': {**std_dict, **label_stds},   # Combine feature and label stds
        'log_transformed_cols': log_transformed_cols,
        'indicator_cols_added': indicator_cols_added,
        'all_cat_features': all_cat_features_base, # Base categorical features
        'all_num_features': all_num_features_final, # Final numerical features (incl. indicators)
        'prefix_cat_cols': prefix_cat_cols, # Specific columns used in prefix tensors
        'prefix_num_cols': prefix_num_cols,
        'suffix_cat_cols': suffix_cat_cols, # Specific columns used in suffix tensors
        'suffix_num_cols': suffix_num_cols,
        'time_label_features': time_label_features, # Specific time labels
        'activity_col': ACTIVITY_COL,
        'case_id_col': CASE_ID_COL,
        'timestamp_col': TIMESTAMP_COL,
        'padding_length': win_size,
        'pad_id': 0, # Standard padding ID
        'eos_token_int': eos_token_int, # Store the integer used for EOS/END
        'num_activities_incl_eos': activity_cardinality + 1, # Includes mapped activities + EOS
        'activity_vocab_size_incl_pad': num_activities_with_special # Includes mapped + EOS + Pad
        # Note: Add outcome info if used
    }
    # Save metadata
    with open(metadata_output_path, 'wb') as f:
        pickle.dump(final_metadata, f)
    print("Metadata saved.")

    print(f"--- Modified Data Preprocessing Finished in {(time.time() - total_start_time)/60:.2f} minutes ---")
    # Return the final tuples if needed elsewhere, otherwise just rely on saved files
    # return train_data_final, val_data_final, test_data_final, final_metadata


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the input CSVs exist before running
    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(TEST_CSV_PATH):
        print(f"Error: Input CSV files not found. Please ensure")
        print(f"'{TRAIN_CSV_PATH}' and '{TEST_CSV_PATH}' exist.")
        print(f"These should be the outputs of the author's create_benchmarks.py step.")
    else:
        run_preprocessing_sutran_like(
            TRAIN_CSV_PATH,
            TEST_CSV_PATH,
            TENSOR_PATH,
            METADATA_PATH,
            validation_split=VALIDATION_SPLIT,
            window_size=WINDOW_SIZE,
            apply_log_transform=APPLY_LOG_TRANSFORM
        )