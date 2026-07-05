"""
Feature-engineering pipeline for the Seattle Airbnb price prediction project.

Takes the cleaned listings dataframe produced by ``clean.py`` (price column
already numeric, NaNs preserved, no rows dropped) and returns a model-ready
feature dataframe.

No rows are dropped here — row eligibility (which listings belong to train vs
test) is owned entirely by ``common.py`` (Person B).  Every row that enters
``build_features`` receives a complete, imputed feature vector.

Each transformation is a small private helper so that individual steps are
independently testable.  Design decisions come from the first-pass exploration
in ``notebooks/tabular_explore.ipynb``; key choices are:

- Clip maximum_nights, bedrooms, bathrooms, estimated_revenue_l365d, and
  reviews_per_month at their train-set 99.9th percentile to remove implausible
  extreme values without losing rows.  Clip thresholds are computed from
  training rows only to prevent test leakage.
- Recover ``bathrooms`` from ``bathrooms_text`` (10.1% vs 0.2% missing).
- Leave categorical columns as clean strings — one-hot / target encoding
  happens separately, right before model training, not here.
- All learned statistics (medians, group medians, category counts, clip
  thresholds) are computed from training rows only.  Pass ``train_ids`` from
  ``common.load_split()`` to activate this; omitting it uses all rows (leaky).
- The regression target (log_price) is owned by common.py — it is NOT included
  in the output of this pipeline.
"""

import ast

import numpy as np
import pandas as pd


# ── Step 1 — clip outliers using train-set quantiles ─────────────────────────

def _clip_outliers(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Clip high-end outliers using train-set 99.9th-percentile thresholds.

    Clipping is applied to all rows (train and test) but the threshold values
    are computed from training rows only, preventing test-set leakage.  Rows
    are never dropped — clipping preserves the row while capping the value.
    """
    clip_cols = [
        "maximum_nights",
        "bedrooms",
        "bathrooms",
        "estimated_revenue_l365d",
        "reviews_per_month",
    ]
    for col in clip_cols:
        if col not in df.columns:
            continue
        cap = df.loc[train_mask, col].quantile(0.999)
        n_clipped = int((df[col] > cap).sum())
        df[col] = df[col].clip(upper=cap)
        if n_clipped:
            print(f"{col:<30} clipped {n_clipped:>3} rows above {cap:.2f} (train 99.9th pct)")

    return df


# ── Step 2 — hotel room flag ─────────────────────────────────────────────────

def _add_hotel_room_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_hotel_room binary flag — created here and dropped in step 17.

    Person B's common.py caps prices at the 99th percentile (~$1,717), which
    excludes all 51 hotel-room listings (~$40k), so is_hotel_room is always 0
    in both train and test (zero variance).  The flag is listed in _COLS_TO_DROP
    and removed before any model sees it.
    """
    if "room_type" in df.columns:
        df["is_hotel_room"] = (df["room_type"] == "Hotel room").astype(int)
    return df


# ── Step 3 — impute host-profile cluster ─────────────────────────────────────

def _impute_host_profile(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Impute the small number of rows (~35 in Seattle) missing host-profile fields.

    Uses train-set medians for numeric columns and conservative sentinel values
    for categorical ones.
    """
    for col in ("host_listings_count", "host_total_listings_count"):
        if col in df.columns:
            median = df.loc[train_mask, col].median()
            df[col] = df[col].fillna(median)

    if "host_verifications" in df.columns:
        df["host_verifications"] = df["host_verifications"].fillna("[]")
    if "host_has_profile_pic" in df.columns:
        df["host_has_profile_pic"] = df["host_has_profile_pic"].fillna("f")
    if "host_identity_verified" in df.columns:
        df["host_identity_verified"] = df["host_identity_verified"].fillna("f")

    return df


# ── Step 4 — recover bathrooms from bathrooms_text ───────────────────────────

def _recover_bathrooms(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Fill missing bathrooms from bathrooms_text; add is_shared_bath flag.

    bathrooms_text has <0.2% missing vs 10.1% for the numeric bathrooms column.
    A regex extracts the leading number (e.g. '1.5 baths' → 1.5).  Rows that
    still cannot be parsed fall back to the train-set median.
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
        bath_median = df.loc[train_mask, "bathrooms"].median()
        df["bathrooms"] = df["bathrooms"].fillna(bath_median)

    return df


# ── Step 5 — impute bedrooms ──────────────────────────────────────────────────

def _impute_bedrooms(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Add bedrooms_missing flag; impute via room_type group medians (train only),
    then train overall median."""
    if "bedrooms" not in df.columns:
        return df

    df["bedrooms_missing"] = df["bedrooms"].isna().astype(int)

    if "room_type" in df.columns:
        train_group_medians = (
            df.loc[train_mask]
            .groupby("room_type")["bedrooms"]
            .median()
        )
        group_fill = df["room_type"].map(train_group_medians)
        df["bedrooms"] = df["bedrooms"].fillna(group_fill)

    overall_median = df.loc[train_mask, "bedrooms"].median()
    df["bedrooms"] = df["bedrooms"].fillna(overall_median)
    return df


# ── Step 6 — impute beds from accommodates ───────────────────────────────────

def _impute_beds(df: pd.DataFrame) -> pd.DataFrame:
    """Where beds is NaN, substitute accommodates (r=0.74 between the two)."""
    if "beds" in df.columns and "accommodates" in df.columns:
        df["beds"] = df["beds"].fillna(df["accommodates"])
    return df


# ── Step 7 — convert percentage strings to floats ────────────────────────────

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


# ── Step 8 — derive years_as_host ────────────────────────────────────────────

def _derive_years_as_host(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Compute years_as_host from host_since; drop the raw date column."""
    if "host_since" not in df.columns:
        return df

    host_since = pd.to_datetime(df["host_since"], errors="coerce")

    if "last_scraped" in df.columns:
        reference_date = pd.to_datetime(df["last_scraped"], errors="coerce").max()
    else:
        reference_date = pd.Timestamp.now()

    df["years_as_host"] = (reference_date - host_since).dt.days / 365.25
    years_median = df.loc[train_mask, "years_as_host"].median()
    df["years_as_host"] = df["years_as_host"].fillna(years_median)
    df = df.drop(columns=["host_since"])
    return df


# ── Step 9 — impute host_is_superhost ────────────────────────────────────────

def _impute_superhost(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing host_is_superhost with 'f' (conservative: assume not superhost)."""
    if "host_is_superhost" in df.columns:
        df["host_is_superhost"] = df["host_is_superhost"].fillna("f")
    return df


# ── Step 10 — impute host_response_time ──────────────────────────────────────

def _impute_response_time(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing host_response_time with an explicit 'unknown' category."""
    if "host_response_time" in df.columns:
        df["host_response_time"] = df["host_response_time"].fillna("unknown")
    return df


# ── Step 11 — sentinel-fill response and acceptance rates ────────────────────

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


# ── Step 12 — parse amenities ─────────────────────────────────────────────────

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

    df["has_wifi"]                = _flag("wifi")
    df["has_kitchen"]             = _flag("kitchen")
    df["has_ac"]                  = _flag("air conditioning")
    df["has_parking"]             = _flag("parking")
    df["has_elevator"]            = _flag("elevator")
    df["has_dedicated_workspace"] = _flag("dedicated workspace")

    df = df.drop(columns=["amenities"])
    return df


# ── Step 13 — has_reviews flag + impute review scores ────────────────────────

def _add_review_flags(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Add has_reviews; impute retained review score columns with train median;
    reviews_per_month with 0."""
    if "review_scores_rating" in df.columns:
        df["has_reviews"] = df["review_scores_rating"].notna().astype(int)
        rating_median = df.loc[train_mask, "review_scores_rating"].median()
        df["review_scores_rating"] = df["review_scores_rating"].fillna(rating_median)

    for col in ("review_scores_location", "review_scores_value"):
        if col in df.columns:
            col_median = df.loc[train_mask, col].median()
            df[col] = df[col].fillna(col_median)

    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    return df


# ── Step 14 — bucket host listing count ──────────────────────────────────────

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


# ── Step 15 — has_license flag ────────────────────────────────────────────────

def _add_license_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary has_license; drop raw license column (values are not structured)."""
    if "license" in df.columns:
        df["has_license"] = df["license"].notna().astype(int)
        df = df.drop(columns=["license"])
    return df


# ── Step 16 — bucket property_type ───────────────────────────────────────────

def _bucket_property_type(
    df: pd.DataFrame, n_top: int = 8, train_mask: pd.Series = None
) -> pd.DataFrame:
    """Keep the top n_top property_type values (by train frequency); replace the
    long tail with 'Other'."""
    if "property_type" not in df.columns:
        return df

    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)

    top = df.loc[train_mask, "property_type"].value_counts().head(n_top).index
    df["property_type"] = df["property_type"].where(
        df["property_type"].isin(top), other="Other"
    )
    return df


# ── Step 17 — drop redundant and unneeded columns ────────────────────────────

_COLS_TO_DROP = [
    # target — use log_price from common.load_target() instead
    "price",
    # zero-variance in the modeling split: Person B's common.py caps prices at
    # the 99th percentile (~$1,717), which excludes all 51 hotel-room listings
    # (~$40,000). is_hotel_room is therefore always 0 in both train and test.
    "is_hotel_room",
    # min-nights redundant cluster — keep minimum_nights
    "minimum_minimum_nights",
    "minimum_nights_avg_ntm",
    "maximum_minimum_nights",
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
    # already parsed in step 4 (_recover_bathrooms)
    "bathrooms_text",
    # out-of-scope / low-value: no imputation plan and not part of the feature spec
    "has_availability",
]


def _drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant columns, all *_url columns, and listing free-text fields."""
    url_cols = [c for c in df.columns if c.endswith("_url")]
    to_drop = _COLS_TO_DROP + url_cols
    return df.drop(columns=[c for c in to_drop if c in df.columns])


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, train_ids=None) -> pd.DataFrame:
    """Take the cleaned listings dataframe and return a model-ready feature dataframe.

    The caller must have already run ``clean_price_column`` on the dataframe.
    No rows are dropped — row eligibility is owned by ``common.py`` (Person B).
    All categorical columns are left as clean strings; encoding happens
    separately, right before model training.

    The regression target (log_price) is NOT included — load it separately via
    ``common.load_target()``.

    All learned statistics — clip thresholds, imputation medians, group medians,
    category frequency counts — are computed from training rows only when
    ``train_ids`` is provided.

    Parameters
    ----------
    df : pd.DataFrame
        Raw-ish listings dataframe with price already numeric (NaNs preserved).
    train_ids : array-like of int, optional
        Listing IDs belonging to the training split (from ``common.load_split()``).
        When provided, all statistics are computed from these rows only, preventing
        test leakage.  When None, statistics use all rows (leaky — only for
        quick sanity checks).

    Returns
    -------
    pd.DataFrame
        Feature dataframe.  Does not contain ``price`` or the log-price target.
    """
    df = df.copy()

    if train_ids is not None and "id" in df.columns:
        train_mask = df["id"].isin(train_ids)
    else:
        train_mask = pd.Series(True, index=df.index)

    df = _clip_outliers(df, train_mask)                        # 1
    df = _add_hotel_room_flag(df)                              # 2
    df = _impute_host_profile(df, train_mask)                  # 3
    df = _recover_bathrooms(df, train_mask)                    # 4
    df = _impute_bedrooms(df, train_mask)                      # 5
    df = _impute_beds(df)                                      # 6
    df = _convert_pct_strings(df)                              # 7
    df = _derive_years_as_host(df, train_mask)                 # 8
    df = _impute_superhost(df)                                 # 9
    df = _impute_response_time(df)                             # 10
    df = _impute_response_rates(df)                            # 11
    df = _parse_amenities(df)                                  # 12
    df = _add_review_flags(df, train_mask)                     # 13
    df = _bucket_host_listings(df)                             # 14
    df = _add_license_flag(df)                                 # 15
    df = _bucket_property_type(df, train_mask=train_mask)      # 16
    df = _drop_redundant(df)                                   # 17

    return df


# ── Quick sanity check when run directly ─────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from src.tabular.clean import load_listings, clean_price_column  # noqa: E402

    data_path      = project_root / "data" / "raw" / "listings.csv"
    train_ids_path = project_root / "data" / "raw" / "train_ids.npy"

    raw     = load_listings(str(data_path))
    cleaned = clean_price_column(raw)

    train_ids = np.load(train_ids_path) if train_ids_path.exists() else None
    if train_ids is None:
        print("Warning: train_ids.npy not found — fitting stats on all data (leaky)")

    result = build_features(cleaned, train_ids=train_ids)
    print(f"Feature shape: {result.shape}")

    missing = result.isna().sum()
    missing = missing[missing > 0]
    print(f"\nColumns with remaining NaN ({len(missing)}):")
    if len(missing):
        print(missing.to_string())
    else:
        print("  (none — all imputation steps succeeded)")
