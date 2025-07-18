# Import libraries
import pandas as pd

# Import dependencies
import config
from s01_preprocess import preprocess_pipeline_buildings, preprocess_pipeline_land
from s02_data_split import split_data_pipeline


def run_pipelines(dpp=True):
    # Run data preprocessing pipeline
    if dpp:
        preprocess_pipeline_land()
        preprocess_pipeline_buildings()

    # Load cleaned data
    df_house = pd.read_csv(config.HOUSE_DATA_PATH)
    df_apt = pd.read_csv(config.APT_DATA_PATH)
    df_land = pd.read_csv(config.LAND_DATA_PATH)

    # Run data split pipeline for buildings
    split_data_pipeline(df_house, "District", config.CROSS_VAL_HOUSE_PATH)
    split_data_pipeline(df_apt, "District", config.CROSS_VAL_APT_PATH)
    split_data_pipeline(df_land, "District", config.CROSS_VAL_LAND_PATH)

    return None


if __name__ == "__main__":
    run_pipelines(dpp=True)
