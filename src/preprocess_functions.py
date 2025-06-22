import polars as pl
from typing import Union, List


def get_nr_of_groups_polars(df: pl.DataFrame, column_names: Union[str, List[str]]):
    if isinstance(column_names, str):
        column_names = [column_names]

    total_rows = df.height

    groups = (
        df.group_by(column_names)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .with_columns((pl.col("count") / total_rows * 100).round(2).alias("percentage"))
    )

    for row in groups.iter_rows(named=True):
        keys = ", ".join(f"{col}={row[col]}" for col in column_names)
        print(f"{keys}: {row['count']} ({row['percentage']}%)")


def assign_as_zero(df, anchor_col: str, anchor_value, assign_col: str):
    return df.with_columns(
        pl.when((pl.col(anchor_col) == anchor_value) & pl.col(assign_col).is_null())
        .then(0)
        .otherwise(pl.col(assign_col))
        .alias(assign_col)
    )
