from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np


def run_random_forest(folds, test_df):
    """
    Run Random Forest on the provided folds and evaluate performance.

    Args:
        folds (list): List of tuples containing train and validation DataFrames.
        test_df (DataFrame): Test DataFrame for final evaluation.

    Returns:
        results (list): List of dictionaries with RMSE and R2 for each fold.
    """

    # Initialize results list
    results = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        # Define predictors and target
        features = [
            col
            for col in train_df.columns
            if col not in ["PricePerSqm", "City", "strat_col"]
        ]  # adjust if needed

        X_train = train_df[features]
        y_train = train_df["PricePerSqm"]

        X_val = val_df[features]
        y_val = val_df["PricePerSqm"]

        # Optional: scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Fit model
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = rf.predict(X_val_scaled)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        results.append({"fold": fold_idx, "rmse": rmse, "r2": r2})

        print(f"Fold {fold_idx}: RMSE={rmse:.2f}, R2={r2:.2f}")
