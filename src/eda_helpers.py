"""Exploratory data analysis helpers."""

import pandas as pd


def column_summary(df: pd.DataFrame, sample_values: int = 0) -> pd.DataFrame:
    """Return a per-column summary: dtype, null stats, cardinality, and optional samples.

    Useful as a quick first look at a new dataset to spot missing data,
    high-cardinality columns, and get a feel for the actual values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to summarize.
    sample_values : int, default 0
        If greater than 0, include a `sample` column showing up to this many
        unique non-null example values per column.

    Returns
    -------
    pd.DataFrame
        One row per column of `df`, indexed by column name, with columns:
        - dtype: the column's data type
        - n_null: number of missing values
        - pct_null: percentage of missing values (rounded to 1 decimal)
        - n_unique: number of unique values (excluding NaN)
        - sample (optional): list of up to `sample_values` example values
    """
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "n_null": df.isna().sum(),
            "pct_null": (df.isna().mean() * 100).round(1),
            "n_unique": df.nunique(),
        }
    )

    if sample_values > 0:
        summary["sample"] = [df[col].dropna().unique()[:sample_values].tolist() for col in df.columns]

    return summary
