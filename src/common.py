"""
common.py — the shared foundation for the Airbnb price project.
"""

import numpy as np
import pandas as pd

DATA_DIR = "/content/drive/MyDrive/Machine_learning_project/Data/Raw"

LISTINGS_CSV = f"{DATA_DIR}/listings.csv"
TARGET_CSV   = f"{DATA_DIR}/target.csv"
TRAIN_IDS    = f"{DATA_DIR}/train_ids.npy"
TEST_IDS     = f"{DATA_DIR}/test_ids.npy"


def build_target_and_split():
    df = pd.read_csv(LISTINGS_CSV)

    df["price"] = df["price"].replace(r"[\$,]", "", regex=True).astype(float)
    df = df[df["price"].notna()].copy()

    cap = df["price"].quantile(0.99)
    df = df[df["price"] <= cap].copy()

    df["log_price"] = np.log(df["price"])

    from sklearn.model_selection import train_test_split
    train_ids, test_ids = train_test_split(
        df["id"].values,
        test_size=0.2,
        random_state=42,
    )

    df[["id", "price", "log_price"]].to_csv(TARGET_CSV, index=False)
    np.save(TRAIN_IDS, train_ids)
    np.save(TEST_IDS, test_ids)

    print(f"{len(df)} usable listings "
          f"(train {len(train_ids)}, test {len(test_ids)})")
    return df


def load_target():
    """Returns a DataFrame: id, price, log_price. Predict log_price."""
    return pd.read_csv(TARGET_CSV)


def load_split():
    """Returns (train_ids, test_ids) — the fixed listing-id membership."""
    return np.load(TRAIN_IDS), np.load(TEST_IDS)
