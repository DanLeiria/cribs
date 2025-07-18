#########################
# --- DATASET PATHS --- #
#########################
RAW_DATA_PATH = "data/raw/raw-data.csv"
BUILD_DATA_PATH = "data/clean/buildings-data.csv"
LAND_DATA_PATH = "data/clean/land-data.csv"

HOUSE_DATA_PATH = "data/clean/house-data.csv"
APT_DATA_PATH = "data/clean/apartment-data.csv"

#####################
# --- CONSTANTS --- #
#####################
SEED = 42

############################
# --- CROSS-VALIDATION --- #
############################
CROSS_VAL_LAND_PATH = "data/cross-val/folds_land.pkl"
CROSS_VAL_HOUSE_PATH = "data/cross-val/folds_house.pkl"
CROSS_VAL_APT_PATH = "data/cross-val/folds_apt.pkl"

N_SPLITS = 5
TEST_SIZE = 0.2
