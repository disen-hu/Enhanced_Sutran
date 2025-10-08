import pandas as pd
import numpy as np
from Preprocessing.from_log_to_tensors import log_to_tensors
import os
import torch

def preprocess_sepsis(log):
    """Preprocess the Sepsis event log.

    Parameters
    ----------
    log : pandas.DataFrame
        Event log.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log.
    """
    # Attempt to convert timestamp column
    # The format '2014-10-22T11:15:41.000+02:00' is ISO 8601, pandas should handle it well.
    # Using utc=True will convert all timestamps to UTC during parsing.
    # errors='coerce' will turn unparseable timestamps into NaT.
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], errors='coerce', utc=True)

    # Check for NaT values after conversion
    nat_count = log['time:timestamp'].isnull().sum()
    if nat_count > 0:
        print(f"Warning: Found {nat_count} NaT (Not a Time) values in 'time:timestamp' after conversion.")
        print("This might indicate issues with some timestamp data formats in your CSV.")
        print("Rows with NaT timestamps will be problematic for time-based operations.")
        # Example: print first 5 rows where timestamp conversion failed
        # print("Problematic rows (first 5):")
        # print(log[log['time:timestamp'].isnull()].head())

    # If, after coercing errors, the column is entirely NaT, it won't be datetime-like.
    # We only proceed with tz_convert if the column is recognized as datetime.
    # Since utc=True was used in to_datetime, tz_convert might be redundant if all inputs were parseable
    # and had timezone info, or if they were naive and assumed UTC.
    # However, if there were mixed valid timezones, utc=True handles the conversion.
    if pd.api.types.is_datetime64_any_dtype(log['time:timestamp']):
        # If pd.to_datetime with utc=True successfully created a UTC-aware DatetimeIndex,
        # further tz_convert to 'UTC' is generally not needed but also not harmful.
        # If the column became object dtype (e.g., all NaT), this block is skipped.
        pass # Already converted to UTC by utc=True in pd.to_datetime
    elif not log['time:timestamp'].isnull().all(): # Check if not all values are NaT
        print("Warning: 'time:timestamp' column is not uniformly datetime-like after initial conversion, but not all are NaT.")
        print("This could happen with mixed data types or extensive parsing errors not resulting in all NaT.")
    else: # All values are NaT or column is not datetime like for other reasons
        print("Error: 'time:timestamp' column could not be converted to a datetime-like format or consists entirely of NaT values.")
        print("Further time-based processing might fail. Please check the 'time:timestamp' column in your CSV.")
        # Depending on requirements, you might raise an error or drop NaT rows here.
        # For now, we'll allow the script to continue, but sorting and windowing might be affected.

    # Sort by case_id and timestamp
    # Only sort by time if the timestamp column is valid and not all NaT
    if 'time:timestamp' in log.columns and pd.api.types.is_datetime64_any_dtype(log['time:timestamp']) and not log['time:timestamp'].isnull().all():
        log.sort_values(by=['case:concept:name', 'time:timestamp'], inplace=True)
    else:
        print("Warning: Sorting by 'time:timestamp' skipped due to issues with the column. Sorting by 'case:concept:name' only.")
        log.sort_values(by=['case:concept:name'], inplace=True)

    # Handle potential NaNs in numeric columns that will be used as features
    # For simplicity, filling with 0. Other strategies (mean, median, ffill) might be better.
    numeric_cols_to_fill = ['Age', 'CRP', 'LacticAcid', 'Leucocytes']
    for col in numeric_cols_to_fill:
        if col in log.columns:
            log[col] = pd.to_numeric(log[col], errors='coerce').fillna(0)
        else:
            print(f"Warning: Numeric column '{col}' not found in the log for NaN filling.")

    # Ensure categorical features are string type to avoid issues with embedding layers
    # Add any other columns that are categorical and might have mixed types or NaNs
    categorical_cols_to_fill_as_str = ['Diagnose', 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition', 'org:group']
    for col in categorical_cols_to_fill_as_str:
        if col in log.columns:
            log[col] = log[col].astype(str).fillna('Missing') # Fill NaNs with 'Missing' string
        else:
            print(f"Warning: Categorical column '{col}' not found in the log for string conversion.")

    return log

def construct_sepsis_datasets(csv_path):
    df = pd.read_csv(csv_path)
    df = preprocess_sepsis(df)

    # Core column names - these should match your CSV
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name' # This is the activity label

    # Define features based on sepsis_cases.csv columns
    # Adjust these lists based on your specific needs and understanding of the data
    categorical_casefeatures = [] # e.g., ['Diagnose'] if 'Diagnose' is a case-level categorical feature
    if 'Diagnose' in df.columns:
        categorical_casefeatures.append('Diagnose')
    
    num_casefts = [] # e.g., ['Age'] if 'Age' is a case-level numerical feature
    if 'Age' in df.columns:
        num_casefts.append('Age')

    categorical_eventfeatures = []
    event_cat_cols = ['org:resource', 'Action', 'EventOrigin', 'lifecycle:transition', 'org:group']
    for col in event_cat_cols:
        if col in df.columns:
            categorical_eventfeatures.append(col)

    numeric_eventfeatures = []
    event_num_cols = ['CRP', 'LacticAcid', 'Leucocytes']
    for col in event_num_cols:
        if col in df.columns:
            numeric_eventfeatures.append(col)
    
    # Dynamically determine date range and max_days from the data
    if not df[timestamp].isnull().all():
        actual_min_date = df[timestamp].min()
        actual_max_date = df[timestamp].max()
        
        if pd.notna(actual_min_date) and pd.notna(actual_max_date):
            start_date = actual_min_date.strftime('%Y-%m-%d')
            end_date = actual_max_date.strftime('%Y-%m-%d')
            max_days = (actual_max_date - actual_min_date).days
            print(f"Dynamically determined start_date: {start_date}")
            print(f"Dynamically determined end_date: {end_date}")
            print(f"Dynamically determined max_days: {max_days}")
        else:
            print("Warning: Could not determine valid min/max dates from data. Using None for date parameters.")
            start_date = None
            end_date = None
            max_days = None
    else:
        print("Warning: Timestamp column is all NaT or empty. Using None for date parameters.")
        start_date = None
        end_date = None
        max_days = None

    window_size = 17 # Hyperparameter, might need tuning
    log_name = 'Sepsis_data' # Output directory name
    start_before_date = None
    test_len_share = 0.25
    val_len_share = 0.2
    mode = 'workaround' # As used in BPIC19 script
    outcome = None # Define if you have an outcome column for prediction tasks

    print(f"Using case_id: {case_id}")
    print(f"Using timestamp: {timestamp}")
    print(f"Using act_label: {act_label}")
    print(f"Categorical Case Features: {categorical_casefeatures}")
    print(f"Numerical Case Features: {num_casefts}")
    print(f"Categorical Event Features: {categorical_eventfeatures}")
    print(f"Numerical Event Features: {numeric_eventfeatures}")

    # Check if essential columns exist
    required_cols = [case_id, timestamp, act_label]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV.")

    result = log_to_tensors(df,
                            log_name=log_name,
                            start_date=start_date,
                            start_before_date=start_before_date,
                            end_date=end_date,
                            max_days=max_days,
                            test_len_share=test_len_share,
                            val_len_share=val_len_share,
                            window_size=window_size,
                            mode=mode,
                            case_id=case_id,
                            act_label=act_label,
                            timestamp=timestamp,
                            cat_casefts=categorical_casefeatures,
                            num_casefts=num_casefts,
                            cat_eventfts=categorical_eventfeatures,
                            num_eventfts=numeric_eventfeatures,
                            outcome=outcome)

    train_data, val_data, test_data = result

    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    train_tensors_path = os.path.join(output_directory, 'train_tensordataset.pt')
    torch.save(train_data, train_tensors_path)
    print(f"Training data saved to {train_tensors_path}")

    val_tensors_path = os.path.join(output_directory, 'val_tensordataset.pt')
    torch.save(val_data, val_tensors_path)
    print(f"Validation data saved to {val_tensors_path}")

    test_tensors_path = os.path.join(output_directory, 'test_tensordataset.pt')
    torch.save(test_data, test_tensors_path)
    print(f"Test data saved to {test_tensors_path}")

if __name__ == '__main__':
    # Path to your Sepsis CSV file
    sepsis_csv_file_path = r'c:\Users\disen\Desktop\root\autodl-tmp\sutran\SuffixTransformerNetwork-main\sepsis_cases.csv'
    
    # Check if the Preprocessing module and log_to_tensors are accessible
    try:
        from Preprocessing.from_log_to_tensors import log_to_tensors
        print("Successfully imported log_to_tensors.")
    except ImportError as e:
        print(f"Error importing log_to_tensors: {e}")
        print("Please ensure 'Preprocessing/from_log_to_tensors.py' is in the correct path and all dependencies are installed.")
        exit()

    construct_sepsis_datasets(sepsis_csv_file_path)
    print("Sepsis dataset construction finished.")