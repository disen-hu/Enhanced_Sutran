# processdata_sutran_like.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import pickle
from sklearn.preprocessing import StandardScaler
import time
from multiprocessing import Pool, cpu_count

# --- Constants ---
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
END_TOKEN = '<end>' # Sutran uses an END token for activity labels
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, END_TOKEN]

# --- Configuration ---
TRAIN_CSV_PATH = 'BPIC_19_train_initial.csv' # Path to train data
TEST_CSV_PATH = 'BPIC_19_test_initial.csv'   # Path to test data
VALIDATION_SPLIT = 0.2  # Percentage of training data to use for validation
OUTPUT_PATH = 'BPIC_19'  # Output folder name
TENSOR_PATH = os.path.join(OUTPUT_PATH, 'processed_tensors_sutran.pt')  # Saved tensors
METADATA_PATH = os.path.join(OUTPUT_PATH, 'metadata_sutran.pt')  # Saved metadata

# Define columns - ADJUST THESE based on your actual CSV columns
CASE_ID_COL = 'case:concept:name'
ACTIVITY_COL = 'concept:name'
TIMESTAMP_COL = 'time:timestamp'
# Add categorical event and case features if available
CAT_CASE_FTS = [] # e.g., ['case:Spend area text', 'case:Document Type']
CAT_EVENT_FTS = [] # e.g., ['org:resource']
NUM_CASE_FTS = [] # e.g., ['case:Total cost']
NUM_EVENT_FTS = [] # e.g., ['Cumulative net worth (EUR)']

# SuTraN processing options
APPLY_LOG_TRANSFORM = True  # Apply log1p transformation to numeric time features
WINDOW_SIZE = 17  # Max sequence length (like window_size in SuTraN)
SAMPLE_PREFIXES = False  # If True, sample prefixes instead of using all prefixes
MAX_PREFIXES_PER_CASE = 5  # If sampling, how many prefixes per case to sample
MIN_PREFIX_LENGTH = 1  # Minimum prefix length to consider
NUM_WORKERS = max(1, cpu_count() - 1)  # Number of workers for parallel processing


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


def add_time_features(df, case_id, timestamp, activity):
    """Adds SuTraN-like time features."""
    start_time = time.time()
    print("Sorting log...")
    df = sort_log(df, case_id, timestamp)
    
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
    
    print(f"Time features added in {time.time() - start_time:.2f} seconds")
    return df


def treat_categorical_features(train_df, test_df, val_df=None, categorical_columns=None):
    """
    Handle categorical features including the activity labels.
    Maps categories to integer IDs and handles unknown values in test set.
    Similar to SuTraN's missing_val_cat_mapping function.
    """
    start_time = time.time()
    print("Processing categorical features...")
    cardinality_dict = {}
    categorical_mapping_dict = {}
    
    if categorical_columns is None:
        categorical_columns = []
    
    # Always include activity column
    if ACTIVITY_COL not in categorical_columns:
        categorical_columns.append(ACTIVITY_COL)
    
    # Process each categorical column
    for col in tqdm(categorical_columns):
        # Convert to string and find unique values in training data
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        if val_df is not None:
            val_df[col] = val_df[col].astype(str)
        
        # Get unique values from training set
        unique_train_values = list(train_df[col].unique())
        
        # Create mapping dictionary
        # Reserve index 0 for padding
        cat_to_int = {v: i+1 for i, v in enumerate(unique_train_values)}
        
        # Add unknown token for test set values not seen in training
        cat_to_int['<unk>'] = len(cat_to_int) + 1
        
        # Apply mapping to datasets using vectorized operations
        # Convert to numpy arrays for faster processing
        train_values = train_df[col].values
        test_values = test_df[col].values
        
        # Create numpy arrays for the transformed values
        train_transformed = np.zeros(len(train_values), dtype=np.int32)
        test_transformed = np.zeros(len(test_values), dtype=np.int32)
        
        # Apply mapping for training set
        for value, idx in cat_to_int.items():
            train_transformed[train_values == value] = idx
        
        # Apply mapping for test set
        for value, idx in cat_to_int.items():
            test_transformed[test_values == value] = idx
            
        # Handle unknown values in test set
        unknown_idx = cat_to_int['<unk>']
        for i, value in enumerate(test_values):
            if value not in cat_to_int:
                test_transformed[i] = unknown_idx
        
        # Update DataFrames
        train_df[col] = train_transformed
        test_df[col] = test_transformed
        
        # Apply mapping to validation set if provided
        if val_df is not None:
            val_values = val_df[col].values
            val_transformed = np.zeros(len(val_values), dtype=np.int32)
            
            # Apply mapping for validation set
            for value, idx in cat_to_int.items():
                val_transformed[val_values == value] = idx
                
            # Handle unknown values in validation set
            for i, value in enumerate(val_values):
                if value not in cat_to_int:
                    val_transformed[i] = unknown_idx
                    
            val_df[col] = val_transformed
        
        # Record cardinality (add +1 for padding token)
        cardinality_dict[col] = len(cat_to_int) + 1
        categorical_mapping_dict[col] = cat_to_int
    
    print(f"Categorical features processed in {time.time() - start_time:.2f} seconds")
    return train_df, test_df, val_df, cardinality_dict, categorical_mapping_dict


def create_validation_split(train_df, case_id, timestamp, val_split=0.2):
    """
    Create a validation set by taking the most recent cases from the training set.
    Similar to the validation split approach in SuTraN.
    """
    start_time = time.time()
    print("Creating validation split...")
    # Get start time for each case
    case_start_df = train_df.groupby(case_id)[timestamp].min().reset_index()
    case_start_df = case_start_df.sort_values(by=timestamp, ascending=True)
    
    # Determine split point
    total_cases = len(case_start_df)
    val_cases = int(total_cases * val_split)
    train_cases = total_cases - val_cases
    
    # Get case IDs for training and validation
    train_case_ids = case_start_df[case_id].iloc[:train_cases].values
    val_case_ids = case_start_df[case_id].iloc[train_cases:].values
    
    # Split the dataframe - use numpy for faster filtering
    train_case_ids_set = set(train_case_ids)
    val_case_ids_set = set(val_case_ids)
    
    # Use boolean indexing for faster filtering
    train_mask = train_df[case_id].isin(train_case_ids_set)
    val_mask = train_df[case_id].isin(val_case_ids_set)
    
    train_filtered = train_df[train_mask].copy()
    val_filtered = train_df[val_mask].copy()
    
    print(f"Training set: {len(train_filtered)} events, {len(train_case_ids)} cases")
    print(f"Validation set: {len(val_filtered)} events, {len(val_case_ids)} cases")
    print(f"Validation split created in {time.time() - start_time:.2f} seconds")
    
    return train_filtered, val_filtered


def preprocess_numericals(train_df, val_df, test_df, numerical_cols, apply_log=True):
    """
    Standardize numerical features (SuTraN style).
    
    Args:
        train_df, val_df, test_df: DataFrames to process
        numerical_cols: List of numerical columns to standardize
        apply_log: Whether to apply log1p transformation to non-negative columns
    
    Returns:
        Processed DataFrames and standardization parameters
    """
    start_time = time.time()
    print("Processing numerical features...")
    means_dict = {}
    std_dict = {}
    scalers = {}
    
    # Identify columns suitable for log transformation
    log_transformable_cols = []
    if apply_log:
        for col in numerical_cols:
            if ((train_df[col] >= 0).all() and 
                (val_df[col] >= 0).all() and 
                (test_df[col] >= 0).all()):
                log_transformable_cols.append(col)
    
    # Apply log transformation first if requested
    if apply_log:
        for col in log_transformable_cols:
            train_df[col] = np.log1p(train_df[col])
            val_df[col] = np.log1p(val_df[col])
            test_df[col] = np.log1p(test_df[col])
    
    # Standardize each numerical column
    for col in numerical_cols:
        scaler = StandardScaler()
        # Fit on training data
        train_values = train_df[col].values.reshape(-1, 1)
        scaler.fit(train_values)
        
        # Transform all datasets
        train_df[col] = scaler.transform(train_values).flatten()
        val_df[col] = scaler.transform(val_df[col].values.reshape(-1, 1)).flatten()
        test_df[col] = scaler.transform(test_df[col].values.reshape(-1, 1)).flatten()
        
        # Store parameters
        means_dict[col] = scaler.mean_[0]
        std_dict[col] = np.sqrt(scaler.var_)[0]
        scalers[col] = scaler
    
    print(f"Numerical features processed in {time.time() - start_time:.2f} seconds")
    return train_df, val_df, test_df, means_dict, std_dict, scalers, log_transformable_cols


def process_case_batch(args):
    """
    Process a batch of cases for creating prefix-suffix pairs.
    This function will be called by multiple processes.
    """
    case_data, activity_col, min_prefix_length, window_size, sample_prefixes, max_prefixes_per_case = args
    
    result = {
        'prefix_activities': [],
        'prefix_times': [],
        'suffix_activities': [],
        'suffix_times': [],
        'time_labels': [],
        'case_lengths': [],
        'prefix_lengths': [],
        'suffix_lengths': [],
        'prefix_padding_mask': []
    }
    
    for case_id, case_df in case_data.items():
        case_length = len(case_df)
        
        # Skip cases with only one event (can't create prefix-suffix)
        if case_length <= 1:
            continue
        
        # Determine prefix lengths to process
        max_prefix_len = min(case_length-1, window_size)
        prefix_lengths = list(range(min_prefix_length, max_prefix_len+1))
        
        # Sample prefix lengths if needed
        if sample_prefixes and len(prefix_lengths) > max_prefixes_per_case:
            # Either sample evenly or include first and last
            if max_prefixes_per_case >= 3:
                # Always include first and last prefix, then sample the rest
                middle_prefixes = prefix_lengths[1:-1]
                if len(middle_prefixes) > (max_prefixes_per_case - 2):
                    sampled_middle = np.random.choice(
                        middle_prefixes, 
                        size=max_prefixes_per_case-2, 
                        replace=False
                    ).tolist()
                    prefix_lengths = [prefix_lengths[0]] + sorted(sampled_middle) + [prefix_lengths[-1]]
                else:
                    # Not enough middle prefixes, just take what we have
                    prefix_lengths = [prefix_lengths[0]] + middle_prefixes + [prefix_lengths[-1]]
            else:
                # Just random sample
                prefix_lengths = np.random.choice(
                    prefix_lengths, 
                    size=min(max_prefixes_per_case, len(prefix_lengths)), 
                    replace=False
                ).tolist()
                prefix_lengths.sort()  # Sort to maintain order
        
        # Extract data once
        activities = case_df[activity_col].values
        times = case_df[['ts_prev', 'ts_start']].values
        time_labels_data = case_df[['tt_next', 'rtime']].values
        
        # Process each prefix length
        for prefix_len in prefix_lengths:
            # Create tensors efficiently
            
            # Prefix activities
            prefix_act = activities[:prefix_len].copy()
            prefix_act_padded = np.zeros(window_size, dtype=np.int64)
            prefix_act_padded[:len(prefix_act)] = prefix_act
            
            # Prefix times
            prefix_times_data = times[:prefix_len].copy()
            prefix_times_padded = np.zeros((window_size, 2), dtype=np.float32)
            prefix_times_padded[:len(prefix_times_data)] = prefix_times_data
            
            # Suffix activities (add END token at end)
            suffix_act = np.concatenate([
                activities[prefix_len:], 
                [len(np.unique(activities))+1]  # END token
            ])
            suffix_act_padded = np.zeros(window_size, dtype=np.int64)
            suffix_act_padded[:min(len(suffix_act), window_size)] = suffix_act[:min(len(suffix_act), window_size)]
            
            # Suffix times
            suffix_times_data = times[prefix_len:].copy()
            suffix_times_padded = np.zeros((window_size, 2), dtype=np.float32)
            suffix_times_padded[:min(len(suffix_times_data), window_size-1)] = suffix_times_data[:min(len(suffix_times_data), window_size-1)]
            
            # Time labels
            time_labels_padded = np.full((window_size, 2), -100.0, dtype=np.float32)
            suffix_time_labels = time_labels_data[prefix_len:].copy()
            time_labels_padded[:min(len(suffix_time_labels), window_size-1)] = suffix_time_labels[:min(len(suffix_time_labels), window_size-1)]
            # Add zero for END token's time labels
            if min(len(suffix_time_labels), window_size-1) < window_size:
                time_labels_padded[min(len(suffix_time_labels), window_size-1)] = np.array([0.0, 0.0])
            
            # Prefix padding mask (True where padded, SuTraN convention)
            prefix_mask = np.ones(window_size, dtype=bool)
            prefix_mask[:len(prefix_act)] = False
            
            # Store arrays in result
            result['prefix_activities'].append(prefix_act_padded)
            result['prefix_times'].append(prefix_times_padded)
            result['suffix_activities'].append(suffix_act_padded)
            result['suffix_times'].append(suffix_times_padded)
            result['time_labels'].append(time_labels_padded)
            result['case_lengths'].append(case_length)
            result['prefix_lengths'].append(prefix_len)
            result['suffix_lengths'].append(min(len(suffix_act), window_size))
            result['prefix_padding_mask'].append(prefix_mask)
    
    return result


def create_prefix_suffix_tensors_optimized(df, case_id, activity, window_size, 
                                          min_prefix_length=1, sample_prefixes=False, 
                                          max_prefixes_per_case=5, num_workers=4):
    """
    Optimized version to create tensors for prefix-suffix pairs.
    Uses parallel processing and efficient data structures.
    
    Args:
        df: Preprocessed DataFrame
        case_id: Case ID column name
        activity: Activity column name
        window_size: Maximum sequence length
        min_prefix_length: Minimum prefix length to consider
        sample_prefixes: Whether to sample prefixes instead of using all
        max_prefixes_per_case: Max number of prefixes per case when sampling
        num_workers: Number of worker processes for parallelization
    
    Returns:
        Dictionary of prefix-suffix tensors for activities and time features
    """
    start_time = time.time()
    print("Creating prefix-suffix tensors...")
    
    # Prepare data by grouping by case ID
    cases = df[case_id].unique()
    print(f"Processing {len(cases)} cases...")
    
    # Create a dictionary of case data for faster access
    case_data = {}
    for case in tqdm(cases, desc="Preparing case data"):
        case_df = df[df[case_id] == case].copy()
        if len(case_df) > 1:  # Skip single-event cases
            case_data[case] = case_df
    
    # Split cases into batches for parallel processing
    num_workers = min(num_workers, len(case_data))
    case_batches = np.array_split(list(case_data.keys()), num_workers)
    batched_case_data = [
        {k: case_data[k] for k in batch if k in case_data} 
        for batch in case_batches
    ]
    
    # Prepare arguments for parallel processing
    process_args = [
        (batch_data, activity, min_prefix_length, window_size, sample_prefixes, max_prefixes_per_case) 
        for batch_data in batched_case_data
    ]
    
    # Process batches in parallel
    print(f"Processing in parallel with {num_workers} workers...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.map(process_case_batch, process_args)
    else:
        # Process sequentially if only one worker
        results = [process_case_batch(args) for args in process_args]
    
    # Combine results from all processes
    combined_result = {
        'prefix_activities': [],
        'prefix_times': [],
        'suffix_activities': [],
        'suffix_times': [],
        'time_labels': [],
        'case_lengths': [],
        'prefix_lengths': [],
        'suffix_lengths': [],
        'prefix_padding_mask': []
    }
    
    for result in results:
        for key in combined_result:
            combined_result[key].extend(result[key])
    
    # Convert lists to tensors
    tensor_dict = {}
    for key in combined_result:
        if combined_result[key]:
            if key in ['case_lengths', 'prefix_lengths', 'suffix_lengths']:
                tensor_dict[key] = torch.tensor(combined_result[key], dtype=torch.long)
            elif key in ['prefix_padding_mask']:
                tensor_dict[key] = torch.tensor(combined_result[key], dtype=torch.bool)
            elif 'activities' in key:
                tensor_dict[key] = torch.tensor(combined_result[key], dtype=torch.long)
            else:
                tensor_dict[key] = torch.tensor(combined_result[key], dtype=torch.float)
    
    total_pairs = len(tensor_dict['prefix_activities'])
    print(f"Created {total_pairs} prefix-suffix pairs in {time.time() - start_time:.2f} seconds")
    
    return tensor_dict


def run_preprocessing_sutran_like(train_csv_path, test_csv_path, tensor_output_path, metadata_output_path, 
                                 validation_split=0.2, window_size=None):
    """
    Processes raw data similar to SuTraN pipeline principles and saves tensors.
    Optimized for speed.
    """
    total_start_time = time.time()
    print("--- Starting SuTraN-like Data Preprocessing ---")

    # --- Load Data ---
    try:
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
        print(f"Loaded data from {train_csv_path} and {test_csv_path}")
    except FileNotFoundError as e:
        print(f"Error: Cannot find data file {e.filename}. Exiting.")
        exit(1)

    # --- Create output directory ---
    os.makedirs(os.path.dirname(tensor_output_path), exist_ok=True)
    
    # --- Add Time Features ---
    df_train = add_time_features(df_train, CASE_ID_COL, TIMESTAMP_COL, ACTIVITY_COL)
    df_test = add_time_features(df_test, CASE_ID_COL, TIMESTAMP_COL, ACTIVITY_COL)
    
    # --- Create validation split ---
    df_train, df_val = create_validation_split(df_train, CASE_ID_COL, TIMESTAMP_COL, validation_split)
    
    # --- Filter by case length (optional) ---
    if window_size:
        print(f"Filtering cases by maximum length: {window_size}")
        df_train = df_train[df_train['case_length'] <= window_size].reset_index(drop=True)
        df_val = df_val[df_val['case_length'] <= window_size].reset_index(drop=True)
        df_test = df_test[df_test['case_length'] <= window_size].reset_index(drop=True)

    # --- Define Feature Sets ---
    all_cat_features = CAT_CASE_FTS + CAT_EVENT_FTS + [ACTIVITY_COL]
    basic_time_features = ['ts_prev', 'ts_start']
    all_num_features = basic_time_features + NUM_CASE_FTS + NUM_EVENT_FTS
    time_label_features = ['tt_next', 'rtime']
    
    # --- Process Categorical Features ---
    df_train, df_test, df_val, cardinality_dict, cat_mapping_dict = treat_categorical_features(
        df_train, df_test, df_val, all_cat_features
    )
    
    # --- Process Numerical Features ---
    numerical_columns = all_num_features + time_label_features
    df_train, df_val, df_test, means_dict, std_dict, scalers, log_transformed_cols = preprocess_numericals(
        df_train, df_val, df_test, numerical_columns, apply_log=APPLY_LOG_TRANSFORM
    )
    
    # --- Create Tensor Datasets ---
    win_size = window_size or WINDOW_SIZE
    train_tensors = create_prefix_suffix_tensors_optimized(
        df_train, CASE_ID_COL, ACTIVITY_COL, win_size,
        min_prefix_length=MIN_PREFIX_LENGTH,
        sample_prefixes=SAMPLE_PREFIXES,
        max_prefixes_per_case=MAX_PREFIXES_PER_CASE,
        num_workers=NUM_WORKERS
    )
    
    val_tensors = create_prefix_suffix_tensors_optimized(
        df_val, CASE_ID_COL, ACTIVITY_COL, win_size,
        min_prefix_length=MIN_PREFIX_LENGTH,
        sample_prefixes=SAMPLE_PREFIXES, 
        max_prefixes_per_case=MAX_PREFIXES_PER_CASE,
        num_workers=NUM_WORKERS
    )
    
    test_tensors = create_prefix_suffix_tensors_optimized(
        df_test, CASE_ID_COL, ACTIVITY_COL, win_size,
        min_prefix_length=MIN_PREFIX_LENGTH,
        sample_prefixes=SAMPLE_PREFIXES,
        max_prefixes_per_case=MAX_PREFIXES_PER_CASE,
        num_workers=NUM_WORKERS
    )
    
    # --- Save Processed Tensors and Metadata ---
    print(f"Saving processed tensors to {tensor_output_path}...")
    tensor_payload = {
        'train': train_tensors,
        'val': val_tensors,
        'test': test_tensors,
        'window_size': win_size
    }
    torch.save(tensor_payload, tensor_output_path)
    print("Processed tensors saved.")

    print(f"Saving metadata to {metadata_output_path}...")
    metadata = {
        'feature_vocabs': {col: {'map': cat_mapping_dict.get(col, {}), 'size': cardinality_dict.get(col, 0)} 
                          for col in all_cat_features},
        'feature_padding_ids': {col: 0 for col in all_cat_features},
        'scalers': scalers,
        'all_cat_features': all_cat_features,
        'all_num_features': all_num_features,
        'time_label_features': time_label_features,
        'padding_length': win_size,
        'log_transformed_columns': log_transformed_cols,
        'means': means_dict,
        'stds': std_dict,
        # Include specific IDs needed by models/dataloaders
        'pad_id': 0,
        'sos_id': 1,  # Assuming SOS is at index 1
        'eos_id': 2,  # Assuming EOS is at index 2
        'unk_id': 3,  # Assuming UNK is at index 3
        'end_id': 4,  # Assuming END is at index 4
        'activity_col': ACTIVITY_COL,
        'vocab_size': cardinality_dict.get(ACTIVITY_COL, 0)
    }
    
    try:
        torch.save(metadata, metadata_output_path)
        print("Metadata saved.")
    except Exception as e:
        print(f"Error saving metadata (scalers might not be serializable): {e}")
        print("Attempting to save metadata without scalers...")
        metadata.pop('scalers', None)  # Remove scalers if saving fails
        torch.save(metadata, metadata_output_path)
        print("Metadata saved (without scalers).")

    print(f"--- SuTraN-like Data Preprocessing Finished in {(time.time() - total_start_time)/60:.2f} minutes ---")
    return train_tensors, val_tensors, test_tensors, metadata


def create_sutran_style_dataset():
    """
    Main function to create SuTraN-style dataset using the BPIC19 data.
    """
    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(TEST_CSV_PATH):
        print(f"Error: Input CSV files not found. Please make sure {TRAIN_CSV_PATH} and {TEST_CSV_PATH} exist.")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Run preprocessing
    train_tensors, val_tensors, test_tensors, metadata = run_preprocessing_sutran_like(
        TRAIN_CSV_PATH, 
        TEST_CSV_PATH,
        TENSOR_PATH,
        METADATA_PATH,
        validation_split=VALIDATION_SPLIT,
        window_size=WINDOW_SIZE
    )
    
    # Verify the created tensors
    print("\nVerifying saved files...")
    if os.path.exists(TENSOR_PATH) and os.path.exists(METADATA_PATH):
        loaded_tensors = torch.load(TENSOR_PATH)
        loaded_metadata = torch.load(METADATA_PATH)
        
        print(f"Train set: {len(loaded_tensors['train']['prefix_activities'])} prefix-suffix pairs")
        print(f"Validation set: {len(loaded_tensors['val']['prefix_activities'])} prefix-suffix pairs")
        print(f"Test set: {len(loaded_tensors['test']['prefix_activities'])} prefix-suffix pairs")
        print(f"Window size: {loaded_metadata['padding_length']}")
        print(f"Activity vocab size: {loaded_metadata['vocab_size']}")
        print(f"Number of categorical features: {len(loaded_metadata['all_cat_features'])}")
        print(f"Number of numerical features: {len(loaded_metadata['all_num_features'])}")
        
        print("Preprocessing successful!")
        return True
    else:
        print("Error: Files not created successfully.")
        return False


if __name__ == "__main__":
    create_sutran_style_dataset()