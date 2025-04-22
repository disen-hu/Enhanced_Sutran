# E-SUTRAN: BART-based Models for Event Sequence Prediction

This repository contains implementations of BART-like models for predicting subsequent events and timestamps in business process event logs, likely using the BPIC 2019 dataset as a basis. Two main approaches are explored: one using standard learned positional embeddings and another using Rotary Positional Embeddings (RoPE).

## Folder Structure

The repository is organized into two main sub-projects:

* **`Bart_like Sutran/`**: Contains the implementation of a BART-like model, likely using standard learned positional embeddings.
    * `processdata.py`: Script for preprocessing the raw data (e.g., BPIC 2019 CSVs) and saving processed data/vocabulary files (`processed_data.pt`, `vocab_info.pt`).
    * `train.py`: Script for training the standard BART-like model using the preprocessed data. Saves the best model weights (`best_model.pth`) and training checkpoints.
    * `test.py`: Script for evaluating the best saved standard BART-like model (`best_model.pth`) on the test set.
    * `Bart-like Sutran.py`: (Possibly an older, monolithic version of the code - you might want to clarify or remove this if `train.py` contains the model definition).

* **`Bart_ROPE/`**: Contains the implementation of a BART-like model enhanced with Rotary Positional Embeddings (RoPE).
    * `process_data.py`: Script for preprocessing raw data (similar to the one above, but potentially specific naming). Saves `processed_data_rope.pt`, `vocab_info_rope.pt`.
    * `train_Bart_ROPE.py`: Script for training the RoPE BART model using its specific preprocessed data. Saves the best model weights (`rope_best_model.pth`) and checkpoints (`rope_checkpoints/`).
    * `test_rope.py`: Script for evaluating the best saved RoPE BART model (`rope_best_model.pth`) on the test set.

**Note:** Raw data files (e.g., `.csv`) and potentially large processed files (`.pt`, `.pth`) are likely excluded via the `.gitignore` file and are not present in this repository.

## Requirements

You need Python 3 and the following libraries:

* `torch` (PyTorch)
* `pandas`
* `numpy`
* `tqdm`
* `rapidfuzz`

You can typically install them using pip:
```bash
pip install torch pandas numpy tqdm rapidfuzz
