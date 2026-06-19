"""
Builds the tabular feature dataframe from raw Seattle Airbnb listings data and
saves it to ``data/processed/tabular_features.csv`` for reuse by notebooks,
the baseline model, and eventually the combined multi-modal model.

Run this script once (or whenever ``clean.py`` or ``features.py`` changes) to
regenerate the processed file.  Downstream consumers can then simply load the
CSV directly instead of re-running the pipeline each time:

    df = pd.read_csv("data/processed/tabular_features.csv")

Re-run any time the feature pipeline changes, or the saved CSV will be stale.
"""

import sys
from pathlib import Path

# Resolve project root so the script works from any working directory
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from src.tabular.clean import clean_price_column, load_listings  # noqa: E402
from src.tabular.features import build_features  # noqa: E402


def main() -> None:
    raw_path = _project_root / "data" / "raw" / "listings.csv"
    out_path = _project_root / "data" / "processed" / "tabular_features.csv"

    print(f"Loading  {raw_path} ...")
    raw = load_listings(str(raw_path))
    print(f"  raw shape: {raw.shape}")

    cleaned = clean_price_column(raw)
    features = build_features(cleaned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    total_nan = features.isna().sum().sum()
    print(f"\nSaved  → {out_path}")
    print(f"Shape    {features.shape[0]:,} rows × {features.shape[1]} columns")
    print(f"NaN      {'0 — all columns fully imputed' if total_nan == 0 else f'{total_nan} remaining (check pipeline)'}")


if __name__ == "__main__":
    main()
