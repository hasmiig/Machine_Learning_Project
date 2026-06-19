import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_split(
    df: pd.DataFrame,
    id_col: str = "id",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return train and test subsets of df, split on listing ID.
    Parameters
    ----------
    df : pd.DataFrame
        The full listings dataframe (or any dataframe that contains id_col).
    id_col : str
        Column whose unique values define one row per listing.
    test_size : float
        Fraction of listings reserved for the test set (default 0.2 → 80/20).
    random_state : int
        Fixed seed — do not change this unless the whole team agrees.

    Returns
    -------
    train_df, test_df : (pd.DataFrame, pd.DataFrame)
        Subsets of df corresponding to the train and test listing IDs.
    """
    ids = df[id_col].unique()
    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=random_state
    )
    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    test_df = df[df[id_col].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df
