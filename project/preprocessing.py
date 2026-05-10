import pandas as pd


def clean_dataframe(df: pd.DataFrame):

    # Keep numeric only
    df = df.select_dtypes(include=["number"])

    # Fill missing
    df = df.fillna(0)

    return df