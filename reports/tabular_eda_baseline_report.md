# Tabular Modality — EDA & Pipeline Report

**Dataset:** Seattle Inside Airbnb (scraped 2025-09-25)
**Updated:** 2026-07-05

---

## What Changed from the Previous Report (2026-06-20)

The old report was written before the pipeline was finalised. The following items in it are **incorrect**:

| Old report | Actual current state |
|---|---|
| Target is `log1p(price)` (step 20 of pipeline) | Target is `log(price)` from `common.load_target()` — not added by the pipeline |
| Hotel rooms are **dropped** (step 3) | Hotel rooms are **kept** with `is_hotel_room = 1` binary flag |
| Step 1: drop `minimum_nights > 30` | No row drops anywhere in the pipeline |
| Step 2: drop rows with missing price | No row drops anywhere in the pipeline |
| Step 5: drop ~35 rows with incomplete host profile | These rows are **imputed** (`_impute_host_profile`) |
| Pipeline has 20 steps | Pipeline has 17 steps |
| Output: 5,793 rows × 53 columns | No row dropping by pipeline; split files: 4,926 × 52 (train), 1,232 × 52 (test) |
| `src/shared/split.py` and `src/shared/build_multimodal.py` exist | Both files deleted; split owned by `src/shared/common.py` (Person B) |
| No models trained yet | 4 sklearn baselines + 3 PyTorch models trained |

---

## 1. Dataset Overview

**Raw data:** `data/raw/listings.csv` — **6,996 listings × 79 columns**

- 42 of 79 columns have at least one missing value
- 775 listings (11.1%) have no `price` — these are excluded from the shared split but are still processed through the pipeline (they receive feature vectors but are not in either train or test)

### Missing Value Highlights

| Column | Missing % | Handling |
|---|---|---|
| `calendar_updated` | 100% | Dropped unconditionally |
| `neighbourhood` | 40% | Duplicate of `neighbourhood_cleansed`, dropped |
| Review scores (7 cols) | ~12.7% | Always co-missing (no reviews yet); median-imputed |
| `price` | 11.1% | 775 rows excluded from split by `common.py` |
| `bathrooms` (numeric) | 10.1% | Recovered from `bathrooms_text` (covers all 708 rows) |
| `host_response_rate/time` | 15.3% | Sentinel/category imputation |
| `host_profile` cluster | ~0.5% | ~35 rows; train-median imputed |

---

## 2. Price Distribution

Raw price is a string like `"$99.00"`. `clean_price_column()` strips `$` and `,`, converts to float.

| Stat | Value |
|---|---|
| Min | $15 |
| Median | $149 |
| Mean | $571 |
| Max | $50,039 |

The distribution is strongly right-skewed. The **regression target is `log_price = log(price)`** (natural log, not log1p), loaded from `data/raw/target.csv` via `common.load_target()`. The pipeline does **not** compute or add the target — it is always loaded separately.

---

## 3. Key Feature Exploration

### Room Type

| Type | Count | Median price |
|---|---|---|
| Entire home/apt | 5,745 (82%) | $161 |
| Private room | 1,175 (17%) | $76 |
| Hotel room | 51 (0.7%) | ~$40,100 |
| Shared room | 25 | $23 |

Hotel rooms follow block/convention pricing. They are **kept in the pipeline** with an `is_hotel_room` binary flag so the model can learn the pricing discontinuity explicitly.

### Neighbourhood

87 fine-grained neighbourhoods (`neighbourhood_cleansed`) or 17 broader districts (`neighbourhood_group_cleansed`). `neighbourhood_group_cleansed` is dropped in step 17 (redundant with `neighbourhood_cleansed`, which is kept and encoded by the models).

### Amenities

Stored as a JSON-like string list. Average: **43 amenities per listing**. Parsed into `amenities_count` + 6 binary flags: `has_wifi`, `has_kitchen`, `has_ac`, `has_parking`, `has_elevator`, `has_dedicated_workspace`.

### Correlation Clusters (|r| > 0.8)

- `minimum_minimum_nights` / `minimum_nights_avg_ntm` / `minimum_nights` (r > 0.94) → keep `minimum_nights`
- `minimum_maximum_nights` / `maximum_maximum_nights` / `maximum_nights_avg_ntm` (r > 0.92) → keep `maximum_nights`
- `availability_30` / `availability_60` / `availability_90` / `availability_365` (r > 0.83) → keep `availability_365`
- `accommodates` / `beds` / `bedrooms` (r > 0.83) → keep `accommodates` and `bedrooms`; impute `beds` from `accommodates` then drop it

---

## 4. Feature Engineering Pipeline (`src/tabular/features.py`)

`build_features()` runs **17 sequential steps**. **No rows are dropped.** Row eligibility (which listings belong to train vs test) is owned entirely by `src/shared/common.py` (Person B). All learned statistics (clip thresholds, medians, group medians, category counts) are computed from **training rows only** to prevent test leakage.

| Step | Function | Action |
|---|---|---|
| 1 | `_clip_outliers` | Clip 5 columns at train 99.9th percentile: `maximum_nights`, `bedrooms`, `bathrooms`, `estimated_revenue_l365d`, `reviews_per_month` |
| 2 | `_add_hotel_room_flag` | Add `is_hotel_room` binary (1 = Hotel room, 0 = residential STR) |
| 3 | `_impute_host_profile` | Impute ~35 rows missing host-profile fields: train median for `host_listings_count`/`host_total_listings_count`; `'[]'` for `host_verifications`; `'f'` for `host_has_profile_pic`/`host_identity_verified` |
| 4 | `_recover_bathrooms` | Add `is_shared_bath` flag; extract numeric `bathrooms` from `bathrooms_text` regex; fallback to train median |
| 5 | `_impute_bedrooms` | Add `bedrooms_missing` flag; fill by `room_type` group median (train only); fallback to train overall median |
| 6 | `_impute_beds` | Fill missing `beds` from `accommodates` (r = 0.74) |
| 7 | `_convert_pct_strings` | Strip `%`; scale `host_response_rate` / `host_acceptance_rate` to [0, 1] |
| 8 | `_derive_years_as_host` | Compute `years_as_host` from `host_since` relative to latest `last_scraped`; fill NaN with train median; drop `host_since` |
| 9 | `_impute_superhost` | Fill missing `host_is_superhost` → `'f'` (conservative: assume not superhost) |
| 10 | `_impute_response_time` | Fill missing `host_response_time` → `'unknown'` category |
| 11 | `_impute_response_rates` | Fill remaining NaN rates → `-1` sentinel (distinguishable from [0,1]; handled by model encoders) |
| 12 | `_parse_amenities` | Expand amenities string → `amenities_count` + 6 binary flags; drop `amenities` |
| 13 | `_add_review_flags` | Add `has_reviews`; train-median impute review scores; fill `reviews_per_month` → 0 |
| 14 | `_bucket_host_listings` | Derive `host_listing_bucket`: `"1 (solo)"` / `"2-5 (small)"` / `"6+ (professional)"` |
| 15 | `_add_license_flag` | Add `has_license` binary; drop raw `license` column |
| 16 | `_bucket_property_type` | Keep top 8 property types by train frequency; replace long tail with `"Other"` |
| 17 | `_drop_redundant` | Drop 30+ columns: redundant clusters, URL columns, scraping metadata, free-text fields, `price` |

### Output

| Split | Rows | Columns | NaN |
|---|---|---|---|
| `tabular_train.csv` | 4,926 | 52 | 0 |
| `tabular_test.csv` | 1,232 | 52 | 0 |
| `tabular_features.csv` (all raw listings) | 6,996 | 52 | 775 (NaN-price rows, not in any split) |

The **52 output columns** include the `id` column (used for joining targets) plus 51 features. The `price` column is dropped by step 17. The target `log_price` is loaded separately from `data/raw/target.csv`.

---

## 5. Pipeline Validation

### Binary Flag Variance (train set, post-pipeline)

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
| `is_hotel_room` | ~1.0% | |
| `bedrooms_missing` | 0.2% | Near-constant — low variance |

### Categorical Columns (encoding happens in the model layer, not here)

| Column | Unique values | Note |
|---|---|---|
| `neighbourhood_cleansed` | 87 | High-cardinality; target-encoded for Ridge, native for trees, OHE for PyTorch |
| `property_type` | 9 | Top 8 + Other |
| `room_type` | 4 | Entire home / Private / Hotel / Shared |
| `host_listing_bucket` | 3 | solo / small / professional |
| `host_response_time` | 5 | within an hour / few hours / within a day / few days / unknown |
| `host_is_superhost` | 2 | t / f |
| `instant_bookable` | 2 | t / f |
| `host_identity_verified` | 2 | t / f |
| `host_verifications` | 6 | e.g. `['email', 'phone']` |
| `host_has_profile_pic` | 2 | t / f; 99% = t (near-constant) |

---

## 6. Baseline Models — Sklearn (`src/tabular/model.py`)

Four sklearn models in increasing complexity, trained via `src/tabular/run_models.py`:

| Model | Key details |
|---|---|
| **Dummy** | Predicts training-set mean log_price for every row — floor baseline |
| **Ridge (RidgeCV)** | StandardScaler on numerics; median impute + scale on rate cols; OrdinalEncoder on `host_response_time`; OHE on low-cardinality nominals; TargetEncoder on `neighbourhood_cleansed` (87 levels); alpha selected via leave-one-out CV over [0.1, 1, 10, 100, 1000, 10000] |
| **Random Forest** | 300 trees, `max_features='sqrt'`, `min_samples_leaf=5`; OrdinalEncoder on all string categoricals; `-1` rate sentinels kept (tree splits on them as a "no data" signal) |
| **LightGBM** | 500 trees, `learning_rate=0.05`, `num_leaves=63`; native categorical support; L2 regularisation (`reg_lambda=1.0`) |

**Outputs saved to `data/processed/`:**
- `tabular_preds_test.csv` — LightGBM test predictions (id, log_price_pred)
- `tabular_preds_train.csv` — LightGBM train predictions

**5-fold cross-validation** (on train set only, test reserved for final one-time reporting) is implemented in `notebooks/tabular_models.ipynb`.

---

## 7. PyTorch Models — Fusion-Ready (`src/tabular/torch_models.py`)

Three PyTorch models designed for downstream multi-modal fusion. Every model exposes:
- `encode(x: Tensor) → Tensor[B, embed_dim]` — the fusion hook
- `model.embed_dim` — embedding size for the fusion head
- `predict_numpy(X, device) → np.ndarray` — convenience wrapper

Target: `log_price = log(price)`. Use `np.exp()` / `torch.exp()` to recover dollar price.

**`FeatureEncoder`** (stateful, fits on train only):
1. Drop `id` and `neighbourhood_group_cleansed`
2. Bool strings (`'t'`/`'f'`) → float 0/1
3. Rate sentinel (-1) → 0.0 + paired `<col>_missing` binary flag
4. OneHotEncoder on all string categoricals
5. StandardScaler across the full encoded array

| Model | Architecture | `embed_dim` | Role |
|---|---|---|---|
| **LinearTabular** | Single `nn.Linear(n_features → 1)`; `encode()` returns the scaled input | n_features | Differentiable linear baseline; fusion gets a standardised tabular vector |
| **MLPTabular** | Three `[Linear → BatchNorm1d → GELU → Dropout]` blocks (256 → 128 → 64) → head | 64 | Non-linear feature interactions |
| **ResNetTabular** | `Linear(n → 128)` → 4 × residual blocks (4× expansion, LayerNorm) → head | 128 | Differentiable analogue of gradient boosting; each block corrects the residual of all preceding blocks |

**Training:** AdamW + CosineAnnealingLR + early stopping (patience=10, best weights restored via `copy.deepcopy`).

**Outputs saved to `data/processed/` by `run_torch_models.py`:**
- `tabular_preds_torch_test.csv` — ResNetTabular test predictions
- `tabular_embeddings_train.npy` — float32 `[4926, 128]` (ResNetTabular encode on full train)
- `tabular_embeddings_test.npy` — float32 `[1232, 128]`

**Fusion pattern (downstream):**
```python
tab_emb = tabular_model.encode(x_tab)   # [B, 128]
txt_emb = text_model.encode(x_txt)      # [B, D_txt]
img_emb = image_model.encode(x_img)     # [B, D_img]
fused   = torch.cat([tab_emb, txt_emb, img_emb], dim=1)
pred    = fusion_head(fused)
```

---

## 8. Project Status

### Done

- Full EDA of all 79 columns: distributions, correlations, categorical cardinalities, missing-value profiles (`notebooks/tabular_explore.ipynb`, 3-part structure)
- Feature engineering pipeline (`features.py`) — 17 steps, no row drops, 0 NaN in split files
- Clean/load layer (`clean.py`)
- Shared split infrastructure: `src/shared/common.py` (Person B) — `train_ids.npy` / `test_ids.npy` / `target.csv`
- Processed output files: `tabular_train.csv`, `tabular_test.csv`, `tabular_features.csv`
- 4 sklearn baseline models with 5-fold CV in notebook (`model.py`, `run_models.py`)
- 3 PyTorch models with fusion-ready `encode()` interface (`torch_models.py`, `run_torch_models.py`)
- Embeddings saved for fusion: `tabular_embeddings_train.npy`, `tabular_embeddings_test.npy`

### What is Next

1. **Run `run_torch_models.py`** and record val/test RMSE for all three PyTorch models. Compare against sklearn baselines.

2. **Text modality baseline.** Once Person C has text features/embeddings, define the shared fusion interface (embedding size, id alignment).

3. **Image modality baseline.** Same as above for Person D.

4. **Multi-modal fusion.** Concatenate `tabular_embeddings_*.npy` with text and image embeddings; train a fusion head (`nn.Linear` or small MLP) on the concatenated vector. Use the same train/test split contract (`train_ids.npy` / `test_ids.npy`).

5. **Consider dropping near-constant flags** before fusion: `has_wifi` (99%), `host_has_profile_pic` (99%), `bedrooms_missing` (0.2%) contribute almost no signal.

6. **Hyperparameter tuning** (optional, after fusion baseline): LightGBM `num_leaves`, `n_estimators`; ResNetTabular `d_model`, `n_blocks`, `dropout`.
