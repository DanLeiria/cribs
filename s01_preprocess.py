# Import libraries
import polars as pl

# Import dependencies
import config
from src.preprocess_functions import get_nr_of_groups_polars, assign_as_zero
from src.logger import PyLogger

# Setup logger
logger = PyLogger(log_to_file=False, file_path="cribs")


def preprocess_pipeline():
    """
    Data preprocessing pipeline.
    Gets the raw data and cleans it,
    The cleaned data is stored in the
    same directory as raw data.
    """
    # Load data as dataframe
    df_raw = pl.read_csv(config.RAW_DATA_PATH)

    # Log raw data load
    nrow_raw = df_raw.height
    logger.info(f"Loaded raw data with {nrow_raw} rows")

    # Get "true" area
    df = df_raw.with_columns(
        pl.when(pl.col("Type") == "Land")
        .then(pl.col("TotalArea"))
        .otherwise(pl.col("LivingArea"))
        .alias("AreaAssigned")
    )

    # Get "true" rooms
    df = df.with_columns(
        pl.when(
            pl.col("TotalRooms").is_not_null()
            & pl.col("NumberOfBedrooms").is_not_null()
        )
        .then(pl.col("TotalRooms"))
        .when(pl.col("TotalRooms").is_not_null())
        .then(pl.col("TotalRooms"))
        .when(pl.col("NumberOfBedrooms").is_not_null())
        .then(pl.col("NumberOfBedrooms"))
        .otherwise(None)  # or a default like 0
        .alias("RoomsAssigned")
    )

    # Remove "empty" columns
    df = df.select(
        pl.exclude(
            [
                "GrossArea",
                "TotalArea",
                "LivingArea",
                "HasParking",
                "Floor",
                "EnergyEfficiencyLevel",
                "PublishDate",
                "NumberOfBedrooms",
                "TotalRooms",
                "NumberOfWC",
                "ConservationStatus",
                "ElectricCarsCharging",
                "LotSize",
                "BuiltArea",
            ]
        )
    )

    # Remove small amount of real estate types
    preserved_types = ["Apartment", "House", "Land"]
    df = df.filter(pl.col("Type").is_in(preserved_types))

    # Remove districts with small samples
    # get_nr_of_groups_polars(df, ["District", "Type"])

    removed_district = [
        "Bragança",
        "Beja",
        "Ilha de Santa Maria",
        "Viseu",
        "Ilha de São Miguel",
        "Ilha de Porto Santo",
        "Z - Fora de Portugal",
        "Ilha Terceira",
        "Ilha da Madeira",
        "Ilha do Faial",
        "Ilha das Flores",
    ]
    df = df.filter(~pl.col("District").is_in(removed_district))

    # Remove missing data of assigned area
    df = df.filter(pl.col("AreaAssigned").is_not_null())

    # Assign zero to a few columns
    df = assign_as_zero(df, "Type", "Land", "ConstructionYear")
    df = assign_as_zero(df, "Type", "Land", "RoomsAssigned")
    df = assign_as_zero(df, "Type", "Land", "NumberOfBathrooms")

    # If garage is missing assign to False
    df = df.with_columns(pl.col("Garage").fill_null(False).alias("Garage"))

    # Remove all leftovers
    df = df.drop_nulls()

    # Check the districts again
    # get_nr_of_groups_polars(df, ["District", "Type"])

    # Log resulting cleaned data
    nrow_clean = df.height
    logger.info(
        f"Preprocessed data with {nrow_clean} rows - {100 * round(nrow_clean / nrow_raw, ndigits=2)}% of the initial dataset."
    )

    # Save preprocessed file
    df.write_csv(config.CLEANED_DATA_PATH, separator=",")


# Run the script
if __name__ == "__main__":
    preprocess_pipeline()
