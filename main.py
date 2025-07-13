# Import libraries
import pandas as pd

# Import dependencies
import config
from s01_preprocess import preprocess_pipeline
from s02_data_split import split_data_pipeline


def run_pipelines(dpp=True):
    # Run data preprocessing pipeline
    if dpp:
        preprocess_pipeline()

    # Load cleaned data
    df_land = pd.read_csv(config.LAND_DATA_PATH)

    split_data_pipeline(df_land, "District")

    return None


if __name__ == "__main__":
    run_pipelines()
