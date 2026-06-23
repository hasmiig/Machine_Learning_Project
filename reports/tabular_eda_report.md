# Tabular Modality — EDA & Pipeline Report

**Dataset:** Seattle Inside Airbnb (scraped 2025-09-25)
**Date:** 2026-06-20

---

## What the Notebook Does — Step by Step

### 1. Setup & Data Loading
The notebook adds the project root to `sys.path` so `src/` imports work, then loads `src/tabular/clean.py` to read `listings.csv`. The raw dataset is **6,996 listings × 79 columns**.

### 2. Missing Values & Data Types
42 of 79 columns have missing data. The worst offenders:

| Column | Missing % | Note |
|---|---|---|
| `calendar_updated` | 100% | Drop unconditionally |
| `neighbourhood` | 40% | Duplicate of `neighbourhood_cleansed`, drop |
| Review scores (7 cols) | ~12.7% | Always missing together (no reviews yet) |
| `price` | 11.1% | 775 rows — drop from training |
| `bathrooms` (numeric) | 10.1% | Recoverable from `bathrooms_text` (covers all 708 rows) |
| `host_response_rate/time` | 15.3% | Sentinel imputation |

### 3. Price Column Cleaning
Raw price is a string like `"$99.00"`. `clean_price_column()` strips `$` and `,`, converts to float.

| Stat | Value |
|---|---|
| Min | $15 |
| Median | $149 |
| Mean | $571 |
| Max | $50,039 |

The distribution is strongly right-skewed. A **log1p transform** is used as the regression target.

### 4. Key Feature Exploration

**Room type breakdown:**

| Type | Count | Median price |
|---|---|---|
| Entire home/apt | 5,745 (82%) | $161 |
| Private room | 1,175 (17%) | $76 |
| Hotel room | 51 (0.7%) | ~$40,100 |
| Shared room | 25 | $23 |

Hotel rooms follow block/convention pricing and are **dropped from modeling**.

**Neighbourhood:** 87 fine-grained neighbourhoods (`neighbourhood_cleansed`) or 17 broader districts (`neighbourhood_group_cleansed`). Downtown mean price (~$3,200) is inflated by hotel rooms; the reliable signal is the **median** (~$177).

**Amenities:** Stored as a JSON-like string list. Average is **43 amenities per listing**. Parsed into `amenities_count` + 6 binary flags.

**Correlation analysis — redundant clusters found (|r| > 0.8):**
- `minimum_minimum_nights` / `minimum_nights_avg_ntm` / `minimum_nights` (r > 0.94)
- `minimum_maximum_nights` / `maximum_maximum_nights` / `maximum_nights_avg_ntm` (r > 0.92)
- `availability_30` / `availability_60` / `availability_90` / `availability_365` (r > 0.83)
- `accommodates` / `beds` / `bedrooms` (r > 0.83)

One feature per cluster is kept; the rest are dropped.

**Minimum nights:** 6.1% of listings require >30 nights (monthly rentals). These are **excluded** from the nightly price model.

### 5. Feature Engineering Pipeline (`src/tabular/features.py`)

`build_features()` runs 20 sequential steps:

| Step | Action |
|---|---|
| 1 | Drop `minimum_nights > 30` |
| 2 | Drop rows with missing price |
| 3 | Drop Hotel room listings |
| 4 | Clip outliers at 99.9th percentile (price, max_nights, bedrooms, bathrooms, revenue, reviews/month) |
| 5 | Drop ~35 rows with incomplete host profile cluster |
| 6 | Recover `bathrooms` from `bathrooms_text`; add `is_shared_bath` flag |
| 7 | Add `bedrooms_missing` flag; impute bedrooms by room_type group median |
| 8 | Impute `beds` from `accommodates` (r = 0.74) |
| 9 | Convert `host_response_rate` / `host_acceptance_rate` from "85%" strings to floats |
| 10 | Derive `years_as_host` from `host_since`; drop raw date column |
| 11 | Impute missing `host_is_superhost` → `'f'` (conservative) |
| 12 | Impute missing `host_response_time` → `'unknown'` category |
| 13 | Impute missing response/acceptance rates → `-1` sentinel |
| 14 | Parse amenities → `amenities_count` + 6 binary flags |
| 15 | Add `has_reviews` flag; median-impute review scores; fill `reviews_per_month` → 0 |
| 16 | Bucket host listing count (solo / small / professional) |
| 17 | Add `has_license` binary flag; drop raw license column |
| 18 | Bucket `property_type` (keep top 8, group rest as "Other") |
| 19 | Drop 30+ redundant columns (URL fields, correlated clusters, scraping metadata) |
| 20 | Add `price_log = log1p(price)` as the regression target |

### 6. Pipeline Validation Result

After the full pipeline:

| Metric | Value |
|---|---|
| Rows | 5,793 |
| Columns | 53 |
| NaN values | 0 |

**Price after pipeline:** median $149, max clipped at $40,037, `price_log` std ≈ 0.66 (well-behaved for regression).

**Binary flag variance check:**

| Flag | % = 1 | Note |
|---|---|---|
| `has_wifi` | 99.0% | Near-constant — low variance |
| `has_kitchen` | 93.9% | |
| `has_reviews` | 91.6% | |
| `has_parking` | 91.5% | |
| `has_license` | 88.0% | |
| `has_dedicated_workspace` | 68.8% | |
| `has_ac` | 45.7% | |
| `has_elevator` | 11.4% | |
| `is_shared_bath` | 9.2% | |
| `bedrooms_missing` | 0.2% | Near-constant — low variance |

**Categorical columns remaining (11) — encoding needed before training:**

| Column | Unique values | Top value |
|---|---|---|
| `neighbourhood_cleansed` | 87 | Broadway (6%) |
| `property_type` | 9 | Entire home (28%) |
| `neighbourhood_group_cleansed` | 17 | Other neighborhoods (21%) |
| `room_type` | 3 | Entire home/apt (83%) |
| `host_listing_bucket` | 3 | 1 (solo) (40%) |
| `host_response_time` | 5 | within an hour (81%) |
| `host_is_superhost` | 2 | t (56%) |
| `instant_bookable` | 2 | f (71%) |
| `host_identity_verified` | 2 | t (89%) |
| `host_verifications` | 6 | ['email', 'phone'] (80%) |
| `host_has_profile_pic` | 2 | t (99%) — near-constant |

---

## Project Status

### Done
- Full EDA of all 79 columns: distributions, correlations, categorical cardinalities, geographic price map, missing-value profiles
- Feature engineering pipeline (`build_features()`) — 20 steps, produces 5,793 × 53, 0 NaNs
- Clean/load layer (`clean.py`)
- Shared infrastructure: `src/shared/split.py` (train/test split) and `src/shared/build_multimodal.py`
- Processed output files: `data/processed/tabular_features.csv`, `data/processed/listings_multimodal.csv`

### Current Results
No model trained yet. The pipeline is validated and ready. The 5,793-row modeling-eligible set represents **84.0% of the raw 6,996 listings**.

### What is Next

1. **Train a baseline model.** `src/tabular/baseline_model.py` is currently a stub. Implement:
   - Linear regression as a sanity-check baseline
   - Random forest or gradient boosting (XGBoost / LightGBM) for a real baseline

2. **Encode categorical features before training.** Recommended approach:
   - `neighbourhood_cleansed` (87 levels) → target encoding
   - `neighbourhood_group_cleansed` (17 levels) → one-hot for a quick baseline
   - Use **only one** of the two neighbourhood columns (they are hierarchically redundant)
   - `room_type`, `property_type`, `host_listing_bucket`, `host_response_time` → one-hot

3. **Evaluate on the shared split.** Use `src/shared/split.py` on the 5,793-row set so tabular, text, and image teams evaluate on identical listings.

4. **Consider dropping near-constant flags.** `has_wifi` (99%) and `bedrooms_missing` (0.2%) and `host_has_profile_pic` (99%) contribute almost no signal — candidates for removal before training.

5. **Decide on neighbourhood granularity.** `neighbourhood_cleansed` (87 groups) gives more signal; `neighbourhood_group_cleansed` (17 districts) is simpler and avoids overfitting. Pick one.

6. **Multi-modal fusion.** Once all three modality baselines are trained, combine predictions (late fusion or a stacking layer).
