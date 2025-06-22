# Import libraries
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# Import dependencies
import config
from src.logger import PyLogger


def split_data_pipeline(df, strat_col, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Use the 'District' column as the stratification key
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df[strat_col])):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        print(f"Fold {fold + 1}")
        print(f"Train districts:\n{train_df['District'].value_counts()}")
        print(f"Test districts:\n{test_df['District'].value_counts()}")
        print("-" * 30)
