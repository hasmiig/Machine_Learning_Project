import pandas as pd


def load_listings(path: str) -> pd.DataFrame:
    """Load the raw Inside Airbnb listings CSV from the given file path."""
    return pd.read_csv(path, low_memory=False)


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Strip currency symbols and commas from the price column and convert to float.

    Handles values like '$1,200.00' → 1200.0. Rows where the price cannot be
    parsed are set to NaN.
    """
    df = df.copy()
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df
