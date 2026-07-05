"""
Builds the tabular feature dataframe from raw Seattle Airbnb listings data and
saves three files to ``data/processed/``:

  tabular_features.csv   — all 6,996 raw rows with features (no row drops)
  tabular_train.csv      — train rows only  (use this for model training)
  tabular_test.csv       — test rows only   (use this for evaluation)

Row eligibility is owned entirely by common.py (Person B): train_ids.npy and
test_ids.npy define exactly which rows belong to each split.  No rows are
dropped in the feature pipeline.

The fixed train/test split (train_ids.npy / test_ids.npy) must already exist
in data/raw/ — run common.py once to generate them if they are missing.

Re-run this script any time clean.py or features.py changes.
"""

import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.tabular.clean import clean_price_column, load_listings  # noqa: E402
from src.tabular.features import build_features                  # noqa: E402


def main() -> None:
    raw_path       = _project_root / "data" / "raw" / "listings.csv"
    train_ids_path = _project_root / "data" / "raw" / "train_ids.npy"
    test_ids_path  = _project_root / "data" / "raw" / "test_ids.npy"
    out_all        = _project_root / "data" / "processed" / "tabular_features.csv"
    out_train      = _project_root / "data" / "processed" / "tabular_train.csv"
    out_test       = _project_root / "data" / "processed" / "tabular_test.csv"

    if not train_ids_path.exists() or not test_ids_path.exists():
        raise FileNotFoundError(
            "train_ids.npy / test_ids.npy not found in data/raw/ — "
            "run common.py once to generate the split."
        )

    train_ids = np.load(train_ids_path)
    test_ids  = np.load(test_ids_path)
    print(f"Split loaded: {len(train_ids):,} train IDs / {len(test_ids):,} test IDs")

    print(f"\nLoading {raw_path} ...")
    raw     = load_listings(str(raw_path))
    cleaned = clean_price_column(raw)
    print(f"  raw shape: {raw.shape}")

    print("\nBuilding features (all learned stats from train rows only) ...")
    features = build_features(cleaned, train_ids=train_ids)

    # Split into train / test using the shared IDs
    train_feat = features[features["id"].isin(train_ids)].reset_index(drop=True)
    test_feat  = features[features["id"].isin(test_ids)].reset_index(drop=True)

    out_all.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_all,   index=False)
    train_feat.to_csv(out_train, index=False)
    test_feat.to_csv(out_test,  index=False)

    train_nan = train_feat.isna().sum().sum()
    test_nan  = test_feat.isna().sum().sum()
    print(f"\nSaved:")
    print(f"  {out_all.name:<25} {features.shape[0]:>5,} rows × {features.shape[1]} columns")
    print(f"  {out_train.name:<25} {len(train_feat):>5,} rows × {train_feat.shape[1]} columns  NaN: {train_nan}")
    print(f"  {out_test.name:<25} {len(test_feat):>5,} rows × {test_feat.shape[1]} columns  NaN: {test_nan}")

    unassigned = features[
        ~features["id"].isin(train_ids) & ~features["id"].isin(test_ids)
    ]
    if len(unassigned):
        print(f"\nNote: {len(unassigned)} rows in tabular_features.csv are outside the shared split "
              f"(no price or above common.py's 99th-pct cap — not used for modeling).")


if __name__ == "__main__":
    main()
