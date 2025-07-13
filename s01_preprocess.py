# Import libraries
import polars as pl

# Import dependencies
import config
from src.preprocess_functions import get_nr_of_groups_polars, assign_as_zero
from src.logger import PyLogger

# Setup logger
logger = PyLogger(log_to_file=True, file_path="cribs")


def preprocess_pipeline_buildings():
    """
    Data preprocessing pipeline for buildings.
    Gets the raw data and cleans it,
    The cleaned data is stored in the
    same directory as raw data.
    """

    ################################
    # -- Load data as dataframe -- #
    ################################
    df_raw = pl.read_csv(config.RAW_DATA_PATH)

    # Log raw data load
    nrow_raw = df_raw.height
    logger.info(f"Loaded raw data with {nrow_raw} rows")

    #########################
    # -- Get "true" area -- #
    #########################
    """This section processes the raw data areas' variables to assign the correct area"""
    df = df_raw.with_columns(
        pl.when(pl.col("Type") == "Land")
        .then(pl.col("TotalArea"))
        .otherwise(pl.col("LivingArea"))
        .alias("AreaAssigned")
    )

    ##########################
    # -- Get "true" rooms -- #
    ##########################
    """This section processes the raw data rooms' variables to assign the correct number of rooms"""
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

    ################################
    # -- Remove "empty" columns -- #
    ################################
    """This section removes columns that are not needed for further analysis because 
    they have a lot of missing values or are not relevant."""
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

    ##################################################
    # -- Remove small amount of real estate types -- #
    ##################################################
    """This section filters the dataset to keep only the relevant real estate types."""
    preserved_types = ["Apartment", "House"]
    df = df.filter(pl.col("Type").is_in(preserved_types))

    #############################################
    # -- Remove districts with small samples -- #
    #############################################
    """This section removes districts with small samples to ensure 
    the dataset is robust for analysis."""

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

    ##############################################
    # -- Remove missing data of assigned area -- #
    ##############################################
    """This section filters out rows where the assigned area is missing or zero."""
    # Remove rows where are is missing
    df = df.filter(pl.col("AreaAssigned").is_not_null())

    # Remove rows where area is zero
    df = df.filter(pl.col("AreaAssigned") != 0)

    ##############################################
    # -- If garage is missing assign to False -- #
    ##############################################
    """This section fills missing values in the 'Garage' column with False,
    indicating that the property does not have a garage."""
    df = df.with_columns(pl.col("Garage").fill_null(False).alias("Garage"))

    ##############################
    # -- Remove all leftovers -- #
    ##############################
    """This section removes all othe rows with null values."""
    df = df.drop_nulls()

    # Check the districts again
    # get_nr_of_groups_polars(df, ["District", "Type"])

    ######################################
    # -- Create Price per Area column -- #
    ######################################
    """This section computes the price per square meter for each property,
    which is a crucial metric for real estate analysis."""
    df = df.with_columns(
        (pl.col("Price") / pl.col("AreaAssigned")).alias("PricePerSqm")
    )

    ###################################
    # -- Remove outliers using IQR -- #
    ###################################
    """This section removes outliers from the dataset based on the Interquartile Range (IQR) method."""

    # Compute group-wise IQR bounds
    iqr_bounds = (
        df.group_by(["District", "Type"])
        .agg(
            [
                pl.col("PricePerSqm").quantile(0.25).alias("Q1"),
                pl.col("PricePerSqm").quantile(0.75).alias("Q3"),
            ]
        )
        .with_columns(  # First compute IQR
            (pl.col("Q3") - pl.col("Q1")).alias("IQR")
        )
        .with_columns(  # Then compute bounds using IQR
            (pl.col("Q1") - 1.5 * pl.col("IQR")).alias("LowerBound"),
            (pl.col("Q3") + 1.5 * pl.col("IQR")).alias("UpperBound"),
        )
        .select(["District", "Type", "LowerBound", "UpperBound"])
    )

    # Join bounds back to original data
    df = df.join(iqr_bounds, on=["District", "Type"], how="left")

    # Filter out outliers using IQR
    df = df.filter(
        (pl.col("PricePerSqm") >= pl.col("LowerBound"))
        & (pl.col("PricePerSqm") <= pl.col("UpperBound"))
    )

    # Optional: drop the helper bound columns
    df = df.drop(["LowerBound", "UpperBound"])

    ###################################
    # -- Give regions per District -- #
    ###################################
    """This section maps each district to its corresponding region, which is useful for regional analysis."""

    # Define a mapping from districts to regions
    district_to_region = {
        "Porto": "Norte",
        "Braga": "Norte",
        "Vila Real": "Norte",
        "Viana do Castelo": "Norte",
        "Aveiro": "Centro",
        "Leiria": "Centro",
        "Coimbra": "Centro",
        "Guarda": "Centro",
        "Castelo Branco": "Centro",
        "Lisboa": "Lisboa",
        "Setúbal": "Lisboa",
        "Santarém": "Centro",
        "Évora": "Alentejo",
        "Portalegre": "Alentejo",
        "Faro": "Algarve",
    }

    df = df.with_columns(
        pl.struct(["District"])
        .map_elements(
            lambda s: district_to_region.get(s["District"], None),
            return_dtype=pl.String,
        )
        .alias("Region")
    )

    #######################################
    # -- Fix energy certificate values -- #
    #######################################
    """This section standardizes the energy certificate values to a consistent format."""

    df = df.with_columns(
        pl.when(pl.col("EnergyCertificate") == "No Certificate")
        .then(pl.lit("NC"))
        .otherwise(pl.col("EnergyCertificate"))
        .alias("EnergyCertificate")
    )

    #######################
    # -- Final dataset -- #
    #######################

    # Log resulting cleaned data
    nrow_clean = df.height
    logger.info(
        f"Preprocessed data of buildings real estate with {nrow_clean} rows - {100 * round(nrow_clean / nrow_raw, ndigits=2)}% of the initial dataset."
    )

    # Save preprocessed dataset (all)
    df.write_csv(config.BUILD_DATA_PATH, separator=",")


def preprocess_pipeline_land():
    """
    Data preprocessing pipeline.
    Gets the raw data from land and cleans it,
    The cleaned data is stored in the
    similar directory as raw data.
    """

    ################################
    # -- Load data as dataframe -- #
    ################################
    df_raw = pl.read_csv(config.RAW_DATA_PATH)

    # Log raw data load
    nrow_raw = df_raw.height
    logger.info(f"Loaded raw data with {nrow_raw} rows")

    ########################
    # -- Pick only land -- #
    ########################
    """This section filters the dataset to keep only the relevant real estate types."""
    df = df_raw.filter(pl.col("Type") == "Land")

    #########################
    # -- Get "true" area -- #
    #########################
    """This section processes the raw data areas' variables to assign the correct area"""
    df = df.with_columns(
        pl.when(pl.col("Type") == "Land")
        .then(pl.col("TotalArea"))
        .otherwise(pl.col("LivingArea"))
        .alias("AreaAssigned")
    )

    ################################
    # -- Remove "empty" columns -- #
    ################################
    """This section removes columns that are not needed for further analysis because 
    they have a lot of missing values or are not relevant."""
    df = df.select(["Price", "District", "City", "AreaAssigned", "Type"])

    ##############################################
    # -- Remove missing data of assigned area -- #
    ##############################################
    """This section filters out rows where the assigned area is missing or zero."""
    # Remove rows where are is missing
    df = df.filter(pl.col("AreaAssigned").is_not_null())

    # Remove rows where area is zero
    df = df.filter(pl.col("AreaAssigned") != 0)

    ##############################
    # -- Remove all leftovers -- #
    ##############################
    """This section removes all othe rows with null values."""
    df = df.drop_nulls()

    # Check the districts again
    # get_nr_of_groups_polars(df, ["District", "Type"])

    ######################################
    # -- Create Price per Area column -- #
    ######################################
    """This section computes the price per square meter for each property,
    which is a crucial metric for real estate analysis."""
    df = df.with_columns(
        (pl.col("Price") / pl.col("AreaAssigned")).alias("PricePerSqm")
    )

    ###################################
    # -- Remove outliers using IQR -- #
    ###################################
    """This section removes outliers from the dataset based on the Interquartile Range (IQR) method."""

    # Compute group-wise IQR bounds
    iqr_bounds = (
        df.group_by(["City"])
        .agg(
            [
                pl.col("PricePerSqm").quantile(0.25).alias("Q1"),
                pl.col("PricePerSqm").quantile(0.75).alias("Q3"),
            ]
        )
        .with_columns(  # First compute IQR
            (pl.col("Q3") - pl.col("Q1")).alias("IQR")
        )
        .with_columns(  # Then compute bounds using IQR
            (pl.col("Q1") - 1.5 * pl.col("IQR")).alias("LowerBound"),
            (pl.col("Q3") + 1.5 * pl.col("IQR")).alias("UpperBound"),
        )
        .select(["City", "LowerBound", "UpperBound"])
    )

    # Join bounds back to original data
    df = df.join(iqr_bounds, on=["City"], how="left")

    # Filter out outliers using IQR
    df = df.filter(
        (pl.col("PricePerSqm") >= pl.col("LowerBound"))
        & (pl.col("PricePerSqm") <= pl.col("UpperBound"))
    )

    # Optional: drop the helper bound columns
    df = df.drop(["LowerBound", "UpperBound"])

    ###################################
    # -- Give regions per District -- #
    ###################################
    """This section maps each district to its corresponding region, which is useful for regional analysis."""

    # Define a mapping from districts to regions
    district_to_region = {
        "Bragança": "Norte",
        "Porto": "Norte",
        "Braga": "Norte",
        "Vila Real": "Norte",
        "Viana do Castelo": "Norte",
        "Viseu": "Centro",
        "Aveiro": "Centro",
        "Leiria": "Centro",
        "Coimbra": "Centro",
        "Guarda": "Centro",
        "Castelo Branco": "Centro",
        "Lisboa": "Lisboa",
        "Setúbal": "Lisboa",
        "Santarém": "Centro",
        "Évora": "Alentejo",
        "Portalegre": "Alentejo",
        "Beja": "Alentejo",
        "Faro": "Algarve",
        "Ilha de Santa Maria": "Açores",
        "Ilha de São Miguel": "Açores",
        "Ilha Terceira": "Açores",
        "Ilha do Faial": "Açores",
        "Ilha das Flores": "Açores",
        "Ilha de Porto Santo": "Madeira",
        "Ilha da Madeira": "Madeira",
    }

    df = df.with_columns(
        pl.struct(["District"])
        .map_elements(
            lambda s: district_to_region.get(s["District"], None),
            return_dtype=pl.String,
        )
        .alias("Region")
    )

    #############################################
    # -- Remove districts with small samples -- #
    #############################################
    """This section removes districts with small samples to ensure 
    the dataset is robust for analysis."""

    removed_district = [
        "Z - Fora de Portugal",
        "Ilha do Faial",
        "Ilha das Flores",
    ]
    df = df.filter(~pl.col("District").is_in(removed_district))

    #######################
    # -- Final dataset -- #
    #######################

    # Log resulting cleaned data
    nrow_clean = df.height
    logger.info(
        f"Preprocessed data of land real estate with {nrow_clean} rows - {100 * round(nrow_clean / nrow_raw, ndigits=2)}% of the initial dataset."
    )

    # Save preprocessed dataset (all)
    df.write_csv(config.LAND_DATA_PATH, separator=",")


def house_dpp_pipeline(df):
    """
    Further clean the house dataset
    """
    # Log the initial dataset
    df = df.filter(pl.col("Type") == "House")

    df.write_csv(config.HOUSE_DATA_PATH, separator=",")


def apt_dpp_pipeline(df):
    """
    Further clean the apartment dataset
    """
    # Log the initial dataset
    df = df.filter(pl.col("Type") == "Apartment")

    df.write_csv(config.APT_DATA_PATH, separator=",")


# Run the script
if __name__ == "__main__":
    # Run the main preprocessing pipeline
    preprocess_pipeline_buildings()
    preprocess_pipeline_land()

    # Separate buildings data by type and save it
    house_dpp_pipeline(df=pl.read_csv(config.BUILD_DATA_PATH))
    apt_dpp_pipeline(df=pl.read_csv(config.BUILD_DATA_PATH))
