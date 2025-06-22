# Import libraries


# Import dependencies
from s01_preprocess import preprocess_pipeline


def run_pipelines(dpp=False):
    # Run data preprocessing pipeline
    if dpp:
        preprocess_pipeline()

    return None
