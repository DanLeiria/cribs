import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os


def plot_comparison():
    # Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the histogram with KDE
    sns.histplot(data=df, x=np.log(df["PricePerSqm"]), kde=True, ax=ax)

    # Add axis labels and title
    ax.set_xlabel("log(PricePerSqm)")
    ax.set_title("Distribution of Price per Sqm in the Dataset")

    # Add vertical lines
    ax.axvline(
        np.log(price_per_sqm),
        color="red",
        linestyle="--",
        label=f"Input: {price_per_sqm:.2f} €/m²",
    )
    ax.axvline(
        np.log(median_price),
        color="darkblue",
        linestyle="--",
        label=f"Median: {median_price:.2f} €/m²",
    )

    # Add caption below the plot
    fig.text(
        0.5,
        -0.05,
        "Figure 1: Distribution of log-transformed price per square meter in the dataset.",
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
    )

    # Show legend and plot
    ax.legend()
    plt.tight_layout()
    plt.show()
