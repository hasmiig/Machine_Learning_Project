"""
Build the shared multi-modal CSV that aligns all three modalities on the same
5,793 modeling-eligible listing IDs.

Run this script once after save_features.py has produced tabular_features.csv.
Re-run any time the tabular pipeline changes (which changes the eligible ID set).

Output
------
data/processed/listings_multimodal.csv — one row per eligible listing:

  Shared
  ------
  id                     listing identifier — used by get_train_test_split
  price / price_log      regression target (kept so every teammate can compute
                         the same evaluation metrics)

  Tabular modality (all 51 feature columns from the pipeline)
  ------------------------------------------------------------
  accommodates, bedrooms, bathrooms, room_type, property_type,
  neighbourhood_cleansed, latitude, longitude, years_as_host,
  amenities_count, has_wifi, has_kitchen, has_ac, has_parking,
  has_elevator, has_dedicated_workspace, review_scores_rating, ...

  Text modality
  -------------
  name                   listing title
  description            full listing description (1.7% missing → empty string)
  neighborhood_overview  host-written neighbourhood blurb (40.3% missing → empty string)

  Image modality
  --------------
  picture_url            URL of the main listing photo (100% present)
  listing_url            Airbnb listing page URL (useful for scraping extra images)

Usage by teammates
------------------
  from src.shared.split import get_train_test_split
  import pandas as pd

  df = pd.read_csv("data/processed/listings_multimodal.csv")
  train, test = get_train_test_split(df)

  # tabular team
  tabular_features = [c for c in df.columns
                      if c not in ("name", "description", "neighborhood_overview",
                                   "picture_url", "listing_url")]
  train_tab = train[tabular_features]

  # text team
  train_text = train[["id", "name", "description", "neighborhood_overview", "price_log"]]

  # image team
  train_img = train[["id", "picture_url", "price_log"]]
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

import pandas as pd  # noqa: E402


_TEXT_IMAGE_COLS = [
    "id",
    # text modality
    "name",
    "description",
    "neighborhood_overview",
    # image modality
    "picture_url",
    "listing_url",
]


def main() -> None:
    tab_path = _project_root / "data" / "processed" / "tabular_features.csv"
    raw_path = _project_root / "data" / "raw" / "listings.csv"
    out_path = _project_root / "data" / "processed" / "listings_multimodal.csv"

    if not tab_path.exists():
        raise FileNotFoundError(
            f"{tab_path} not found — run src/tabular/save_features.py first."
        )

    print(f"Loading tabular features from {tab_path} ...")
    tab = pd.read_csv(tab_path)
    print(f"  {tab.shape[0]:,} rows × {tab.shape[1]} columns")

    print(f"\nLoading raw listings from {raw_path} ...")
    raw = pd.read_csv(raw_path, low_memory=False, usecols=_TEXT_IMAGE_COLS)
    print(f"  {raw.shape[0]:,} rows")

    # Fill missing text with empty string so teammates don't need to handle NaN
    for col in ("description", "neighborhood_overview"):
        if col in raw.columns:
            raw[col] = raw[col].fillna("")

    print("\nMerging on listing id (inner join — keeps only modeling-eligible rows) ...")
    merged = tab.merge(raw, on="id", how="inner")
    print(f"  {merged.shape[0]:,} rows × {merged.shape[1]} columns after merge")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"\nSaved → {out_path}")
    print(f"Shape   {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    print(f"\nColumns added for text/image teammates:")
    for col in _TEXT_IMAGE_COLS[1:]:
        if col in merged.columns:
            pct = (merged[col] != "").mean() * 100 if merged[col].dtype == object else merged[col].notna().mean() * 100
            print(f"  {col:<25} {pct:.1f}% non-empty")


if __name__ == "__main__":
    main()
