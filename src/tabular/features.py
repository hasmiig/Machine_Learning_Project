"""
Feature-engineering pipeline for the Seattle Airbnb price prediction project.

Takes the cleaned listings dataframe produced by ``clean.py`` (price column
already numeric, NaNs preserved, no rows dropped) and returns a model-ready
feature dataframe suitable for training a nightly price regression model.

Each transformation is a small private helper so that individual steps are
independently testable.  Design decisions come from the first-pass exploration
in ``notebooks/tabular_explore.ipynb``; key choices are:

- Drop listings with minimum_nights > 30 (monthly rentals price differently).
- Drop Hotel room listings (room_type == "Hotel room") — they follow hotel/
  block pricing (median ~$40k) rather than residential nightly STR pricing
  and would dominate the regression loss if included.
- Drop listings with missing price; drop rows with price < $10 (placeholder
  entries — 0 rows in Seattle, minimum is $15); clip price, maximum_nights,
  bedrooms, bathrooms, estimated_revenue_l365d, and reviews_per_month at their
  99.9th percentile to remove implausible extreme values without losing the
  rows entirely.  Final modeling-eligible set: ~5,800 rows out of 6,996 raw.
- Recover ``bathrooms`` from ``bathrooms_text`` (10.1% vs 0.2% missing).
- Leave categorical columns as clean strings — one-hot / target encoding
  happens separately, right before model training, not here.
- Log1p-transform ``price`` into ``price_log`` as the regression target.
"""

import ast

import numpy as np
import pandas as pd


# ── Step 1 — filter long-stay listings ───────────────────────────────────────

def _filter_long_stay(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where minimum_nights > 30 (monthly rentals, not nightly stays)."""
    if "minimum_nights" not in df.columns:
        return df
    return df[df["minimum_nights"] <= 30].reset_index(drop=True)


# ── Step 2 — filter missing target ───────────────────────────────────────────

def _filter_missing_price(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where price is NaN (no target, can't train on them)."""
    if "price" not in df.columns:
        return df
    return df.dropna(subset=["price"]).reset_index(drop=True)


# ── Step 3 — filter Hotel room listings ──────────────────────────────────────

def _filter_hotel_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Hotel room listings — they follow hotel/block pricing (median ~$40k)
    rather than residential nightly STR pricing and would dominate the loss."""
    if "room_type" not in df.columns:
        return df
    before = len(df)
    df = df[df["room_type"] != "Hotel room"].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"room_type (Hotel room):   dropped {dropped} rows")
    return df


# ── Step 4 — handle outliers ─────────────────────────────────────────────────

def _handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop implausibly low prices; clip high-end outliers in price, maximum_nights,
    bedrooms, bathrooms, estimated_revenue_l365d, and reviews_per_month.

    ``price`` below $10 is dropped outright (likely broken/placeholder listings).
    The remaining columns are clipped (not dropped) at the 99.9th percentile —
    each has a small number of implausible extreme values far beyond its own
    75th percentile, almost certainly data errors or leftover platform defaults
    rather than real listings.  Clipping preserves each row's other feature
    data while preventing extreme values from distorting the model.

    Note: ``bedrooms`` and ``bathrooms`` are clipped here on their raw (already-
    present) values; missing values in those columns are handled separately by
    ``_impute_bedrooms`` and ``_recover_bathrooms``, which run afterward.
    """
    if "price" in df.columns:
        before = len(df)
        df = df[df["price"] >= 10].reset_index(drop=True)
        print(f"price:                    dropped {before - len(df)} rows below $10")

    clip_cols = [
        "price",
        "maximum_nights",
        "bedrooms",
        "bathrooms",
        "estimated_revenue_l365d",
        "reviews_per_month",
    ]
    for col in clip_cols:
        if col not in df.columns:
            continue
        cap = df[col].quantile(0.999)
        n_clipped = int((df[col] > cap).sum())
        df[col] = df[col].clip(upper=cap)
        if n_clipped:
            print(f"{col:<30} clipped {n_clipped:>3} rows above {cap:.2f} (99.9th pct)")

    return df


# ── Step 5 — drop rows missing the host-profile cluster ──────────────────────

def _drop_incomplete_host_profile_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the small number of rows (~35 in Seattle) missing the host-profile cluster
    (host_listings_count, host_verifications, host_has_profile_pic,
    host_identity_verified) rather than imputing four columns for a
    negligible amount of data loss."""
    cols = [
        "host_listings_count",
        "host_verifications",
        "host_has_profile_pic",
        "host_identity_verified",
    ]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    return df.dropna(subset=present).reset_index(drop=True)


# ── Step 6 — recover bathrooms from bathrooms_text ───────────────────────────

def _recover_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing bathrooms from bathrooms_text; add is_shared_bath flag.

    bathrooms_text has <0.2% missing vs 10.1% for the numeric bathrooms column in Seattle.
    A regex extracts the leading number (e.g. '1.5 baths' → 1.5).  Rows that
    still cannot be parsed fall back to the overall median.
    """
    if "bathrooms_text" in df.columns:
        df["is_shared_bath"] = (
            df["bathrooms_text"]
            .str.lower()
            .str.contains("shared", na=False)
            .astype(int)
        )
    else:
        df["is_shared_bath"] = 0

    if "bathrooms" in df.columns and "bathrooms_text" in df.columns:
        missing = df["bathrooms"].isna()
        if missing.any():
            extracted = (
                df.loc[missing, "bathrooms_text"]
                .astype(str)
                .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
            )
            df.loc[missing, "bathrooms"] = pd.to_numeric(extracted, errors="coerce")

    if "bathrooms" in df.columns:
        df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    return df


# ── Step 7 — impute bedrooms ──────────────────────────────────────────────────

def _impute_bedrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Add bedrooms_missing flag; impute via room_type group medians, then overall median."""
    if "bedrooms" not in df.columns:
        return df

    df["bedrooms_missing"] = df["bedrooms"].isna().astype(int)

    if "room_type" in df.columns:
        group_medians = df.groupby("room_type")["bedrooms"].transform("median")
        df["bedrooms"] = df["bedrooms"].fillna(group_medians)

    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    return df


# ── Step 8 — impute beds from accommodates ───────────────────────────────────

def _impute_beds(df: pd.DataFrame) -> pd.DataFrame:
    """Where beds is NaN, substitute accommodates (r=0.74 between the two)."""
    if "beds" in df.columns and "accommodates" in df.columns:
        df["beds"] = df["beds"].fillna(df["accommodates"])
    return df


# ── Step 9 — convert percentage strings to floats ────────────────────────────

def _convert_pct_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip % signs; scale host_response_rate / host_acceptance_rate to [0, 1]."""
    for col in ("host_response_rate", "host_acceptance_rate"):
        if col not in df.columns:
            continue
        df[col] = (
            pd.to_numeric(
                df[col].astype(str).str.replace("%", "", regex=False).str.strip(),
                errors="coerce",
            )
            / 100
        )
    return df


# ── Step 10 — derive years_as_host ────────────────────────────────────────────

def _derive_years_as_host(df: pd.DataFrame) -> pd.DataFrame:
    """Compute years_as_host from host_since; drop the raw date column."""
    if "host_since" not in df.columns:
        return df

    host_since = pd.to_datetime(df["host_since"], errors="coerce")

    if "last_scraped" in df.columns:
        reference_date = pd.to_datetime(df["last_scraped"], errors="coerce").max()
    else:
        reference_date = pd.Timestamp.now()

    df["years_as_host"] = (reference_date - host_since).dt.days / 365.25
    df["years_as_host"] = df["years_as_host"].fillna(df["years_as_host"].median())
    df = df.drop(columns=["host_since"])
    return df


# ── Step 11 — impute host_is_superhost ────────────────────────────────────────

def _impute_superhost(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing host_is_superhost with 'f' (conservative: assume not superhost)."""
    if "host_is_superhost" in df.columns:
        df["host_is_superhost"] = df["host_is_superhost"].fillna("f")
    return df


# ── Step 12 — impute host_response_time ───────────────────────────────────────

def _impute_response_time(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing host_response_time with an explicit 'unknown' category."""
    if "host_response_time" in df.columns:
        df["host_response_time"] = df["host_response_time"].fillna("unknown")
    return df


# ── Step 13 — sentinel-fill response and acceptance rates ────────────────────

def _impute_response_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaN rate values with -1 sentinel after pct conversion.

    -1 is outside [0, 1] and distinguishable from real values — fine for
    tree-based models.  Do NOT feed these columns as-is into a linear model
    without adding a separate is_missing indicator and imputing the rate column.
    """
    for col in ("host_response_rate", "host_acceptance_rate"):
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    return df


# ── Step 14 — parse amenities ─────────────────────────────────────────────────

def _parse_amenity_list(val) -> list:
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _parse_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """Expand amenities string into count + targeted binary flags; drop original."""
    if "amenities" not in df.columns:
        return df

    lists = df["amenities"].apply(_parse_amenity_list)
    df["amenities_count"] = lists.apply(len)

    def _flag(pattern: str) -> pd.Series:
        p = pattern.lower()
        return lists.apply(lambda lst: int(any(p in item.lower() for item in lst)))

    # "Wifi" and "Pocket wifi" both satisfy the wifi check via substring match
    df["has_wifi"]                = _flag("wifi")
    df["has_kitchen"]             = _flag("kitchen")
    df["has_ac"]                  = _flag("air conditioning")
    df["has_parking"]             = _flag("parking")
    df["has_elevator"]            = _flag("elevator")
    df["has_dedicated_workspace"] = _flag("dedicated workspace")

    df = df.drop(columns=["amenities"])
    return df


# ── Step 15 — has_reviews flag + impute review scores ────────────────────────

def _add_review_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add has_reviews; impute all retained review score columns with median; reviews_per_month with 0."""
    if "review_scores_rating" in df.columns:
        df["has_reviews"] = df["review_scores_rating"].notna().astype(int)
        df["review_scores_rating"] = df["review_scores_rating"].fillna(
            df["review_scores_rating"].median()
        )

    for col in ("review_scores_location", "review_scores_value"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "reviews_per_month" in df.columns:
        # listings with no reviews have 0 reviews per month, not the global average
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    return df


# ── Step 16 — bucket host listing count ──────────────────────────────────────

def _bucket_host_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Derive host_listing_bucket (solo / small / professional) string column."""
    if "calculated_host_listings_count" not in df.columns:
        return df

    def _bucket(x):
        if pd.isna(x):
            return None
        elif x == 1:
            return "1 (solo)"
        elif x <= 5:
            return "2-5 (small)"
        else:
            return "6+ (professional)"

    df["host_listing_bucket"] = df["calculated_host_listings_count"].apply(_bucket)
    return df


# ── Step 17 — has_license flag ────────────────────────────────────────────────

def _add_license_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary has_license; drop raw license column (values are not structured)."""
    if "license" in df.columns:
        df["has_license"] = df["license"].notna().astype(int)
        df = df.drop(columns=["license"])
    return df


# ── Step 18 — bucket property_type ───────────────────────────────────────────

def _bucket_property_type(df: pd.DataFrame, n_top: int = 8) -> pd.DataFrame:
    """Keep the top n_top property_type values; replace the long tail with 'Other'."""
    if "property_type" not in df.columns:
        return df
    top = df["property_type"].value_counts().head(n_top).index
    df["property_type"] = df["property_type"].where(
        df["property_type"].isin(top), other="Other"
    )
    return df


# ── Step 19 — drop redundant and unneeded columns ────────────────────────────

_COLS_TO_DROP = [
    # min-nights redundant cluster — keep minimum_nights
    "minimum_minimum_nights",
    "minimum_nights_avg_ntm",
    "maximum_minimum_nights",   # follow-up: missed from this cluster in original spec
    # max-nights redundant cluster — keep maximum_nights
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "maximum_nights_avg_ntm",
    # availability redundant cluster — keep availability_365
    "availability_30",
    "availability_60",
    "availability_90",
    # beds (imputed from accommodates; accommodates is kept)
    "beds",
    # always empty in this snapshot
    "calendar_updated",
    # constant in Seattle snapshot (always 'city scrape') — adds no information
    "source",
    # review sub-scores with near-zero variance / high collinearity
    # (keep review_scores_rating, review_scores_location, review_scores_value)
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    # scraping metadata
    "scrape_id",
    "last_scraped",
    "calendar_last_scraped",
    # host identifiers and free-text (not useful as tabular features)
    "host_id",
    "host_name",
    "neighbourhood",
    "host_neighbourhood",
    "host_about",
    "host_location",
    "neighborhood_overview",
    # listing free text — used by the text modality, not tabular
    "description",
    "name",
    # date columns already consumed or not useful as raw strings
    "first_review",
    "last_review",
    # already parsed in step 6 (_recover_bathrooms)
    "bathrooms_text",
    # out-of-scope / low-value: no imputation plan and not part of the feature spec
    "has_availability",
]


def _drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant columns, all *_url columns, and listing free-text fields."""
    url_cols = [c for c in df.columns if c.endswith("_url")]
    to_drop = _COLS_TO_DROP + url_cols
    return df.drop(columns=[c for c in to_drop if c in df.columns])


# ── Step 20 — log-transform the target ───────────────────────────────────────

def _log_transform_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_log = log1p(price); keep original price for result interpretation."""
    if "price" in df.columns:
        df["price_log"] = np.log1p(df["price"])
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Take the cleaned listings dataframe and return a model-ready feature dataframe.

    The caller must have already run ``clean_price_column`` on the dataframe.
    Rows are filtered (long-stay, missing price) but the original dataframe is
    never mutated.  All categorical columns are left as clean strings; encoding
    happens separately, right before model training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw-ish listings dataframe with price already numeric (NaNs preserved).

    Returns
    -------
    pd.DataFrame
        Feature dataframe with ~5.8k rows (Seattle) and clean dtypes, ready for encoding
        and model training.  Contains ``price`` (original) and ``price_log``
        (log1p-transformed target).
    """
    df = df.copy()

    df = _filter_long_stay(df)                    # 1
    df = _filter_missing_price(df)                # 2
    df = _filter_hotel_rooms(df)                  # 3
    df = _handle_outliers(df)                     # 4
    df = _drop_incomplete_host_profile_rows(df)   # 5
    df = _recover_bathrooms(df)                   # 6
    df = _impute_bedrooms(df)                     # 7
    df = _impute_beds(df)                         # 8
    df = _convert_pct_strings(df)                 # 9
    df = _derive_years_as_host(df)                # 10
    df = _impute_superhost(df)                    # 11
    df = _impute_response_time(df)                # 12
    df = _impute_response_rates(df)               # 13
    df = _parse_amenities(df)                     # 14
    df = _add_review_flags(df)                    # 15
    df = _bucket_host_listings(df)                # 16
    df = _add_license_flag(df)                    # 17
    df = _bucket_property_type(df)                # 18
    df = _drop_redundant(df)                      # 19
    df = _log_transform_price(df)                 # 20

    return df


# ── Quick sanity check when run directly ─────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from src.tabular.clean import load_listings, clean_price_column  # noqa: E402

    data_path = project_root / "data" / "raw" / "listings.csv"
    print(f"Loading {data_path} ...")

    raw = load_listings(str(data_path))
    cleaned = clean_price_column(raw)

    print(f"Raw shape:     {cleaned.shape}")
    result = build_features(cleaned)
    print(f"Feature shape: {result.shape}")

    print(f"\nDtypes:\n{result.dtypes.to_string()}")

    missing = result.isna().sum()
    missing = missing[missing > 0]
    print(f"\nColumns with remaining NaN ({len(missing)}):")
    if len(missing):
        print(missing.to_string())
    else:
        print("  (none — all imputation steps succeeded)")
