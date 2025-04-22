# processdata_sutran_like.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import pickle
from sklearn.preprocessing import StandardScaler

# --- Constants ---
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
# MASK_TOKEN = '<mask>' # Optional, depending on model use
UNK_TOKEN = '<unk>'
END_TOKEN = '<end>' # Sutran uses an END token for activity labels
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, END_TOKEN]

# --- Configuration ---
TRAIN_CSV_PATH = 'BPIC_19_train_initial.csv' # Assumes this is the training split
TEST_CSV_PATH = 'BPIC_19_test_initial.csv'   # Assumes this is the test split
# Consider adding a validation split logic if needed
OUTPUT_TENSOR_FILE = 'processed_tensors_sutran.pt' # Saved tensors
OUTPUT_METADATA_FILE = 'metadata_sutran.pt'       # Saved vocab, scalers etc.

# Define columns - ADJUST THESE based on your actual CSV columns
CASE_ID_COL = 'case:concept:name'
ACTIVITY_COL = 'concept:name'
TIMESTAMP_COL = 'time:timestamp'
# Add other categorical or numerical feature columns if available
# CAT_EVENT_FTS = ['Resource', ...]
# NUM_EVENT_FTS = ['Amount', ...]
CAT_EVENT_FTS = []
NUM_EVENT_FTS = []


def sort_log(df, case_id, timestamp):
    """Sorts log by case start time, then by timestamp within case."""
    df[timestamp] = pd.to_datetime(df[timestamp])
    # Sort by timestamp within case first
    df = df.sort_values([case_id, timestamp], ascending=True, kind='mergesort')
    # Then sort cases by their start time
    df['case_start_time'] = df.groupby(case_id)[timestamp].transform('min')
    df = df.sort_values(['case_start_time', case_id, timestamp], ascending=True, kind='mergesort')
    df = df.drop(columns=['case_start_time'])
    return df.reset_index(drop=True)

def add_time_features(df, case_id, timestamp):
    """Adds Sutran-like time features."""
    df = sort_log(df, case_id, timestamp)
    df['case_length'] = df.groupby(case_id, sort=False)[timestamp].transform('size')

    # Time delta from previous event (ts_prev)
    df['prev_timestamp'] = df.groupby(case_id, sort=False)[timestamp].shift(1)
    df['ts_prev'] = (df[timestamp] - df['prev_timestamp']).dt.total_seconds()
    df['ts_prev'] = df['ts_prev'].fillna(0.0) # First event has 0 delta

    # Time delta until next event (tt_next)
    df['next_timestamp'] = df.groupby(case_id, sort=False)[timestamp].shift(-1)
    df['tt_next'] = (df['next_timestamp'] - df[timestamp]).dt.total_seconds()
    # Last event's tt_next will be NaN - keep it for now, handle during label creation

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
    return df

def run_preprocessing_sutran_like(train_csv_path, test_csv_path, tensor_output_path, metadata_output_path):
    """
    Processes raw data similar to Sutran pipeline principles and saves tensors.
    """
    print("--- Starting Sutran-like Data Preprocessing ---")

    # --- Load Data ---
    try:
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
        print(f"Loaded data from {train_csv_path} and {test_csv_path}")
    except FileNotFoundError as e:
        print(f"Error: Cannot find data file {e.filename}. Exiting."); exit(1)

    # --- Add Time Features ---
    print("Adding time features...")
    df_train = add_time_features(df_train, CASE_ID_COL, TIMESTAMP_COL)
    df_test = add_time_features(df_test, CASE_ID_COL, TIMESTAMP_COL)

    # --- Define Features ---
    # Activity label is treated separately for vocab
    all_cat_features = CAT_EVENT_FTS # Add CAT_CASE_FTS if you have them
    all_num_features = ['ts_prev', 'ts_start'] + NUM_EVENT_FTS # Base numerical features per event
    # Time labels to be predicted (will also be standardized)
    time_label_features = ['tt_next', 'rtime']

    # --- Handle Categorical Features (Activity + Others) ---
    print("Processing categorical features...")
    feature_vocabs = {}
    feature_padding_ids = {}

    # Build vocabulary for activity from training data
    activity_vocab = sorted(list(df_train[ACTIVITY_COL].astype(str).unique()))
    # Create mapping: Add PAD, SOS, EOS, UNK, END first
    act_event_to_id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    next_id = len(SPECIAL_TOKENS)
    for event in activity_vocab:
        if event not in act_event_to_id:
            act_event_to_id[event] = next_id
            next_id += 1
    act_id_to_event = {i: w for w, i in act_event_to_id.items()}
    act_vocab_size = len(act_event_to_id)
    act_pad_id = act_event_to_id[PAD_TOKEN]
    act_unk_id = act_event_to_id[UNK_TOKEN]
    act_eos_id = act_event_to_id[EOS_TOKEN] # Used for activity labels
    act_end_id = act_event_to_id[END_TOKEN] # Sutran specific end token for labels


    feature_vocabs[ACTIVITY_COL] = {'map': act_event_to_id, 'size': act_vocab_size}
    feature_padding_ids[ACTIVITY_COL] = act_pad_id

    # Process other categorical features (similar logic, simplified: no OOV/missing handling here)
    for col in all_cat_features:
        vocab = sorted(list(df_train[col].astype(str).unique()))
        event_to_id = {PAD_TOKEN: 0} # Reserve 0 for padding
        for i, event in enumerate(vocab):
             event_to_id[event] = i + 1 # Start IDs from 1
        vocab_size = len(event_to_id)
        feature_vocabs[col] = {'map': event_to_id, 'size': vocab_size}
        feature_padding_ids[col] = 0 # Pad ID is 0

    # Apply mapping
    df_train[ACTIVITY_COL] = df_train[ACTIVITY_COL].astype(str).map(lambda x: act_event_to_id.get(x, act_unk_id))
    df_test[ACTIVITY_COL] = df_test[ACTIVITY_COL].astype(str).map(lambda x: act_event_to_id.get(x, act_unk_id))
    for col in all_cat_features:
        mapping = feature_vocabs[col]['map']
        pad_id = feature_padding_ids[col]
        df_train[col] = df_train[col].astype(str).map(lambda x: mapping.get(x, pad_id)) # Map unknowns to pad
        df_test[col] = df_test[col].astype(str).map(lambda x: mapping.get(x, pad_id))

    # --- Handle Numerical Features (Standardization) ---
    print("Processing numerical features...")
    scalers = {}
    all_numeric_to_scale = all_num_features + time_label_features

    # Handle potential NaN in tt_next (last event) before scaling
    df_train['tt_next'] = df_train['tt_next'].fillna(0.0) # Fill last event's tt_next with 0
    df_test['tt_next'] = df_test['tt_next'].fillna(0.0)

    # Apply Log transform similar to Sutran (before scaling) - Optional
    log_transform = True # Set to False to disable
    if log_transform:
        print("Applying log1p transformation...")
        for col in all_numeric_to_scale:
             # Check if column exists and is non-negative before applying log1p
            if col in df_train.columns and (df_train[col] >= 0).all():
                 df_train[col] = np.log1p(df_train[col])
                 if col in df_test.columns and (df_test[col] >= 0).all():
                     df_test[col] = np.log1p(df_test[col])
            else:
                 print(f"  Skipping log1p for column '{col}' (might contain negative values or not exist)")


    # Fit scalers on training data
    for col in all_numeric_to_scale:
        if col in df_train.columns:
            scaler = StandardScaler()
            # Reshape using .values.reshape(-1, 1) before fitting/transforming
            df_train[col] = scaler.fit_transform(df_train[col].values.reshape(-1, 1)).flatten()
            scalers[col] = scaler # Store the fitted scaler
            # Transform test data
            if col in df_test.columns:
                df_test[col] = scaler.transform(df_test[col].values.reshape(-1, 1)).flatten()
            else:
                 df_test[col] = 0 # Or handle missing columns differently
        else:
            print(f"Warning: Column '{col}' not found for scaling.")
            df_test[col] = 0 # Ensure test set has column even if train didn't

    # --- Group Data by Case and Convert to Tensors ---
    print("Grouping data and creating tensors...")
    all_features = [ACTIVITY_COL] + all_cat_features + all_num_features
    label_features = [ACTIVITY_COL] # Activity label needs special handling (shifting, END token)
    time_labels = time_label_features

    def df_to_padded_tensors(df, max_len):
        # Note: Sutran pads differently, often creating tensors directly.
        # This simplified approach pads lists first.
        grouped = df.groupby(CASE_ID_COL)
        all_case_data = []
        for _, group in tqdm(grouped, desc="Padding cases"):
            case_dict = {}
            group_len = len(group)
            pad_len = max(0, max_len - group_len)

            # --- Features ---
            case_dict['activities'] = torch.tensor(
                [act_event_to_id[SOS_TOKEN]] + group[ACTIVITY_COL].tolist() + ([act_pad_id] * pad_len),
                dtype=torch.long
            )[:max_len+1] # Add SOS, pad, truncate

            for col in all_cat_features:
                 pad_val = feature_padding_ids[col]
                 # Assuming no SOS needed for other features
                 case_dict[f'feat_{col}'] = torch.tensor(
                     group[col].tolist() + ([pad_val] * pad_len),
                     dtype=torch.long
                 )[:max_len]

            num_data_list = []
            for col in all_num_features:
                 # Add 0.0 for SOS position's delta features, pad with 0.0
                 data = [0.0] + group[col].tolist() + ([0.0] * pad_len)
                 num_data_list.append(data[:max_len+1]) # Include SOS position
            if num_data_list:
                 case_dict['num_features'] = torch.tensor(num_data_list, dtype=torch.float).t() # Transpose to (len, num_feat)

            # --- Labels ---
            # Activity labels: sequence shifted left, add END token, pad
            act_labels = group[ACTIVITY_COL].tolist()[1:] + [act_end_id] # Shift, add END
            case_dict['activity_labels'] = torch.tensor(
                act_labels + ([act_pad_id] * pad_len),
                dtype=torch.long
            )[:max_len] # Pad/truncate labels

            # Time labels: sequence shifted left, pad with 0 (or ignore index)
            time_labels_list = []
            for col in time_labels:
                 # Add 0.0 for SOS position, pad with 0.0
                 data = [0.0] + group[col].tolist() + ([0.0] * pad_len)
                 time_labels_list.append(data[:max_len+1])
            if time_labels_list:
                 case_dict['time_labels'] = torch.tensor(time_labels_list, dtype=torch.float).t() # (len, num_time_labels)
                 # Shift time labels left to align with prediction task (predict tt_next/rtime *for* event i based on event i-1)
                 # Shape: (len-1, num_time_labels) - targets for output sequence of length len-1
                 case_dict['time_labels_target'] = case_dict['time_labels'][1:, :]


            case_dict['length'] = torch.tensor(group_len + 1, dtype=torch.long) # Length including SOS
            all_case_data.append(case_dict)
        return all_case_data # Returns list of dicts, to be handled by custom collate/Dataset

    # Find max length across both train and test for consistent padding basis
    # Add 1 for SOS token
    max_len_train = df_train.groupby(CASE_ID_COL).size().max() + 1
    max_len_test = df_test.groupby(CASE_ID_COL).size().max() + 1
    global_max_len = max(max_len_train, max_len_test)
    print(f"Max sequence length found (incl. SOS): {global_max_len}")

    # Determine padding length (e.g., use global max or a fixed window size)
    # Sutran uses window_size from percentile - let's use a fixed large enough value or global_max_len
    padding_length = global_max_len # Or set a fixed MAX_SEQ_LENGTH_MODEL here
    print(f"Padding sequences to length: {padding_length}")

    # Process train and test sets
    # Note: This is memory intensive as it creates lists of tensors first
    # Sutran's tensor_creation is likely more memory efficient for large logs
    train_data_list = df_to_padded_tensors(df_train, padding_length)
    test_data_list = df_to_padded_tensors(df_test, padding_length)

    # --- Save Processed Tensors and Metadata ---
    print(f"Saving processed tensors to {tensor_output_path}...")
    # Save as lists of dictionaries - requires custom Dataset/Collate in train/test
    # Saving large lists of tensors can be slow/memory heavy.
    # Consider saving batches or using a more efficient format if performance is an issue.
    tensor_payload = {
        'train': train_data_list,
        'test': test_data_list,
        'padding_length': padding_length
    }
    torch.save(tensor_payload, tensor_output_path)
    print("Processed tensors saved.")

    print(f"Saving metadata to {metadata_output_path}...")
    metadata = {
        'feature_vocabs': feature_vocabs, # Contains maps and sizes for categoricals
        'feature_padding_ids': feature_padding_ids,
        'scalers': scalers, # Contains fitted scalers for numericals
        'all_cat_features': all_cat_features, # List of other categorical cols
        'all_num_features': all_num_features, # List of numerical feature cols
        'time_label_features': time_label_features, # List of time label cols
        'padding_length': padding_length,
        # Include specific IDs needed by models/dataloaders
        'pad_id': act_pad_id,
        'sos_id': act_event_to_id[SOS_TOKEN],
        'eos_id': act_event_to_id[EOS_TOKEN],
        'unk_id': act_unk_id,
        'end_id': act_end_id,
        'activity_col': ACTIVITY_COL,
        'vocab_size': act_vocab_size # Specifically activity vocab size
    }
    # Note: Scalers might not pickle correctly, saving params might be better
    # For simplicity, saving the objects here.
    try:
        torch.save(metadata, metadata_output_path)
        print("Metadata saved.")
    except Exception as e:
        print(f"Error saving metadata (scalers might not be serializable): {e}")
        print("Attempting to save metadata without scalers...")
        metadata.pop('scalers', None) # Remove scalers if saving fails
        torch.save(metadata, metadata_output_path)
        print("Metadata saved (without scalers). You may need to re-fit scalers or save parameters.")


    print("--- Sutran-like Data Preprocessing Finished ---")

if __name__ == "__main__":
    run_preprocessing_sutran_like(
        TRAIN_CSV_PATH,
        TEST_CSV_PATH,
        OUTPUT_TENSOR_FILE,
        OUTPUT_METADATA_FILE
    )
    # Verification
    if os.path.exists(OUTPUT_TENSOR_FILE) and os.path.exists(OUTPUT_METADATA_FILE):
        print("\nVerifying saved files...")
        loaded_tensors = torch.load(OUTPUT_TENSOR_FILE)
        loaded_metadata = torch.load(OUTPUT_METADATA_FILE)
        print(f"Loaded {len(loaded_tensors['train'])} train sequences.")
        print(f"Loaded {len(loaded_tensors['test'])} test sequences.")
        print(f"Padding length: {loaded_metadata['padding_length']}")
        print(f"Activity vocab size: {loaded_metadata['vocab_size']}")
        # print(f"Scalers loaded: {'scalers' in loaded_metadata}")
        print("Preprocessing script finished successfully.")