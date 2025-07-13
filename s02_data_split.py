# Import libraries
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import os
import pickle

# Import dependencies
import config
from src.logger import PyLogger

# Setup logger
logger = PyLogger(log_to_file=True, file_path="cribs")


def split_data_pipeline(
    df,
    strat_col,
    test_size=0.2,
    n_splits=5,
    random_state=42,
    save_path=config.CROSS_VAL_DATA_PATH,
):
    """
    Test size is 20% of the total size
    Number of splits is 5 folds
    Folds stratified by the column "strat_col"
    """
    # Step 1: Stratified train/test split
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[strat_col], random_state=random_state
    )

    # Step 2: Stratified K-Fold on the training/validation set
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_val_df, train_val_df[strat_col])
    ):
        train_fold = train_val_df.iloc[train_idx].copy()
        val_fold = train_val_df.iloc[val_idx].copy()
        folds.append((train_fold, val_fold))

    # Save to a file
    with open(save_path, "wb") as f:
        pickle.dump({"folds": folds, "test_set": test_df}, f)

    logger.info(f"Folds and test set saved to: {os.path.relpath(save_path)}")
    return folds, test_df


if __name__ == "__main__":
    with open(config.CROSS_VAL_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    folds = data["folds"]
    test_set = data["test_set"]

    print(folds)
    print(test_set)
