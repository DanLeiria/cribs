import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import LAND_DATA_PATH


def compare_land(input_dict, df_path=LAND_DATA_PATH):
    """
    Match input land specs with land dataset.
    """

    df = pd.read_csv(df_path)

    # Keywords to skip of the input dictionary
    skip_keywords = ["AreaAssigned", "Price"]

    # Filter dtaframe according to the inputs
    for column, value in input_dict.items():
        # Skip specific columns
        if any(kw in column for kw in skip_keywords):
            continue  # Skip unwanted filters

        if isinstance(value, list):
            # If more than one value in an input index
            df = df[df[column].isin(value)]
        else:
            df = df[df[column] == value]

    if df.empty:
        print("No matches found - check your input values.")
        return None

    print(df.head())

    # Get the area and the price of the input
    area_input = input_dict.get("AreaAssigned", None)
    price_input = input_dict.get("Price", None)

    if not area_input or not price_input:
        print("Missing area or price input for comparison.")
        return None

    # Calculate the price per square meter
    price_per_sqm = price_input / area_input
    print(f"Input Price per Sqm: {price_per_sqm:.2f} EUR/m2")

    # Median price per square meter
    median_price = df["PricePerSqm"].median()

    sns.histplot(data=df, x=np.log(df["PricePerSqm"]), kde=True)
    plt.xlabel("log(PricePerSqm)")
    plt.title("Distribution of Price per Sqm in the Dataset")
    plt.axvline(
        np.log(price_per_sqm),
        color="red",
        linestyle="--",
        label=f"Input: {price_per_sqm:.2f} €/m2",
    )
    plt.axvline(
        np.log(median_price),
        color="darkblue",
        linestyle="--",
        label=f"Median: {median_price:.2f} €/m2",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    return None


if __name__ == "__main__":
    land_dict = {
        "Region": "Algarve",
        "District": "Faro",
        "City": "Faro",
        "AreaAssigned": 2200,  # in square meters
        "Price": 27000,  # in euros
    }

    compare_land(input_dict=land_dict)
