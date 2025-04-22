# preprocess_data_rope.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os

# --- Constants ---
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
MASK_TOKEN = '<mask>' # Kept for consistency, though RoPE BART might not use it explicitly for training task
UNK_TOKEN = '<unk>'   # Added for unknown tokens in test set
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, MASK_TOKEN, UNK_TOKEN]

# --- Configuration ---
TRAIN_CSV_PATH = 'BPIC_19_train_initial.csv'
TEST_CSV_PATH = 'BPIC_19_test_initial.csv'
OUTPUT_DATA_FILE = 'processed_data_rope.pt' # Specific name for RoPE version
OUTPUT_VOCAB_FILE = 'vocab_info_rope.pt'     # Specific name for RoPE version
MAX_SEQ_LENGTH_PREPROCESS = None # Set to an integer (e.g., 100) if you want to truncate during preprocessing

def run_preprocessing(train_csv, test_csv, data_output_path, vocab_output_path, max_len=None):
    """
    Reads raw CSVs, preprocesses, encodes, and saves data and vocab.
    Args:
        train_csv (str): Path to the training CSV file.
        test_csv (str): Path to the test CSV file.
        data_output_path (str): Path to save the processed sequence/delta data.
        vocab_output_path (str): Path to save the vocabulary information.
        max_len (int, optional): If provided, sequences longer than this will be truncated *before* adding SOS/EOS.
    """
    print("--- Starting Data Preprocessing (for RoPE model) ---")

    # --- Load Raw Data ---
    try:
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        print(f"Loaded data from {train_csv} and {test_csv}")
    except FileNotFoundError as e:
        print(f"Error: Cannot find data file {e.filename}. Exiting.")
        exit(1)

    df_train['time:timestamp'] = pd.to_datetime(df_train['time:timestamp'])
    df_test['time:timestamp'] = pd.to_datetime(df_test['time:timestamp'])

    # --- Process Function ---
    def process_group(group_df):
        group_df = group_df.sort_values('time:timestamp')
        events = group_df['concept:name'].tolist()
        times = group_df['time:timestamp'].tolist()
        deltas_sec = [0.0] + [max(0.0, (times[i] - times[i-1]).total_seconds()) for i in range(1, len(times))]
        deltas_log = np.log1p(deltas_sec).tolist()

        if max_len is not None:
            events = events[:max_len]
            deltas_log = deltas_log[:max_len+1] # Deltas have initial 0.0

        # Return deltas corresponding to events (exclude initial 0.0)
        return events, deltas_log[1:] # Length matches events

    # --- Process Training Data ---
    print("Processing training data...")
    grouped_train = df_train.groupby('case:concept:name', group_keys=False)
    train_sequences_raw = []
    train_time_deltas_raw = []
    for _, group in tqdm(grouped_train, desc="Processing train cases"):
        events, deltas = process_group(group)
        train_sequences_raw.append(events)
        train_time_deltas_raw.append(deltas)

    # --- Build Vocabulary ---
    print("Building vocabulary...")
    all_events = set(e for seq in train_sequences_raw for e in seq)
    vocab = SPECIAL_TOKENS + sorted(list(all_events))
    event_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_event = {i: w for w, i in event_to_id.items()}

    pad_id = event_to_id[PAD_TOKEN]
    s_id = event_to_id[SOS_TOKEN]
    eos_id = event_to_id[EOS_TOKEN]
    mask_id = event_to_id[MASK_TOKEN]
    unk_id = event_to_id[UNK_TOKEN]
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Pad ID: {pad_id}, Unk ID: {unk_id}")

    # --- Encode Training Sequences (Add SOS/EOS and corresponding 0 deltas) ---
    print("Encoding training sequences...")
    encoded_train_sequences = []
    encoded_train_deltas = []
    for seq, dts in tqdm(zip(train_sequences_raw, train_time_deltas_raw), total=len(train_sequences_raw), desc="Encoding train"):
        tokens = [s_id] + [event_to_id.get(e, unk_id) for e in seq] + [eos_id]
        deltas_with_sos_eos = [0.0] + dts + [0.0] # Add 0 delta for SOS and EOS
        encoded_train_sequences.append(tokens)
        encoded_train_deltas.append(deltas_with_sos_eos)
        # Simple check:
        if len(tokens) != len(deltas_with_sos_eos):
             print(f"Warning: Length mismatch during train encoding. Seq len {len(seq)}, Final Tokens {len(tokens)}, Final Deltas {len(deltas_with_sos_eos)}")


    # --- Process Test Data ---
    print("Processing test data...")
    grouped_test = df_test.groupby('case:concept:name', group_keys=False)
    test_sequences_raw = []
    test_time_deltas_raw = []
    for _, group in tqdm(grouped_test, desc="Processing test cases"):
        events, deltas = process_group(group)
        test_sequences_raw.append(events)
        test_time_deltas_raw.append(deltas)

    # --- Encode Test Sequences (Add SOS/EOS and corresponding 0 deltas) ---
    print("Encoding test sequences...")
    encoded_test_sequences = []
    encoded_test_deltas = []
    unknown_event_count = 0
    total_test_event_tokens = 0
    for seq, dts in tqdm(zip(test_sequences_raw, test_time_deltas_raw), total=len(test_sequences_raw), desc="Encoding test"):
        encoded_seq_part = []
        for e in seq:
            token_id = event_to_id.get(e, unk_id)
            if token_id == unk_id and e != UNK_TOKEN: # Count actual unknown events
                unknown_event_count += 1
            encoded_seq_part.append(token_id)
            total_test_event_tokens += 1

        tokens = [s_id] + encoded_seq_part + [eos_id]
        deltas_with_sos_eos = [0.0] + dts + [0.0]
        encoded_test_sequences.append(tokens)
        encoded_test_deltas.append(deltas_with_sos_eos)
        if len(tokens) != len(deltas_with_sos_eos):
             print(f"Warning: Length mismatch during test encoding. Seq len {len(seq)}, Final Tokens {len(tokens)}, Final Deltas {len(deltas_with_sos_eos)}")

    if total_test_event_tokens > 0:
        print(f"Found {unknown_event_count} unknown event instances in test set ({unknown_event_count/total_test_event_tokens*100:.2f}%), mapped to UNK token.")

    # --- Save Processed Data ---
    print(f"Saving processed data to {data_output_path}...")
    data_payload = {
        'train_seqs': encoded_train_sequences,
        'train_deltas': encoded_train_deltas,
        'test_seqs': encoded_test_sequences,
        'test_deltas': encoded_test_deltas,
    }
    torch.save(data_payload, data_output_path)
    print("Processed data saved.")

    # --- Save Vocabulary Info ---
    print(f"Saving vocabulary info to {vocab_output_path}...")
    vocab_payload = {
        'event_to_id': event_to_id,
        'id_to_event': id_to_event,
        'vocab_size': vocab_size,
        'pad_id': pad_id,
        's_id': s_id,
        'eos_id': eos_id,
        'mask_id': mask_id,
        'unk_id': unk_id, # Save unk_id as well
    }
    torch.save(vocab_payload, vocab_output_path)
    print("Vocabulary info saved.")

    print("--- Data Preprocessing Finished ---")


if __name__ == "__main__":
    # Create dummy CSV files if needed
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"Creating dummy train CSV: {TRAIN_CSV_PATH}")
        dummy_train_data = {'case:concept:name': ['c1','c1','c2','c2'], 'concept:name': ['A','B','X','Y'], 'time:timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 11:00:00', '2023-01-01 11:10:00'])}
        pd.DataFrame(dummy_train_data).to_csv(TRAIN_CSV_PATH, index=False)
    if not os.path.exists(TEST_CSV_PATH):
        print(f"Creating dummy test CSV: {TEST_CSV_PATH}")
        dummy_test_data = {'case:concept:name': ['c3','c3','c4'], 'concept:name': ['A','C','X'], 'time:timestamp': pd.to_datetime(['2023-01-02 09:00:00', '2023-01-02 09:03:00', '2023-01-02 10:00:00'])}
        pd.DataFrame(dummy_test_data).to_csv(TEST_CSV_PATH, index=False)

    # Run the preprocessing
    run_preprocessing(TRAIN_CSV_PATH, TEST_CSV_PATH, OUTPUT_DATA_FILE, OUTPUT_VOCAB_FILE, max_len=MAX_SEQ_LENGTH_PREPROCESS)

    # Optional: Verify saved files
    if os.path.exists(OUTPUT_DATA_FILE) and os.path.exists(OUTPUT_VOCAB_FILE):
        print("\nVerifying saved files...")
        loaded_data = torch.load(OUTPUT_DATA_FILE)
        loaded_vocab = torch.load(OUTPUT_VOCAB_FILE)
        print(f"Loaded {len(loaded_data['train_seqs'])} train sequences.")
        print(f"Loaded {len(loaded_data['test_seqs'])} test sequences.")
        print(f"Loaded vocab size: {loaded_vocab['vocab_size']}")
        print("Preprocessing script (RoPE version) finished successfully.")
        # Clean up dummy files if they were created
        # if 'dummy' in TRAIN_CSV_PATH: os.remove(TRAIN_CSV_PATH)
        # if 'dummy' in TEST_CSV_PATH: os.remove(TEST_CSV_PATH)
    else:
        print("Error: Output files not found after running preprocessing.")