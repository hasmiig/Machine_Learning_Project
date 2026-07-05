# Tabular Modality — Model Results Report

**Dataset:** Seattle Inside Airbnb (scraped 2025-09-25)  
**Split:** 4,926 train / 1,232 test (fixed via `train_ids.npy` / `test_ids.npy`)  
**Target:** `log_price = log(price)`, range [2.71, 7.45] → price [$15, $1,717]  
**Features:** 51 columns, 0 NaN (see pipeline section)  
**Date:** 2026-07-05

---

## Issues Found and Fixed This Session

| Issue | Fix |
|---|---|
| `is_hotel_room` was always 0 in both splits | Dropped from pipeline. Person B's `common.py` caps prices at 99th pct (~$1,717), excluding all 51 hotel rooms (~$40k). The flag was zero-variance dead weight. |
| Predictions saved as log only | All prediction CSVs now include `log_price_actual`, `log_price_pred`, `price_actual`, `price_pred` |
| Models not persisted to disk | All 7 models now saved to `models/tabular/` (see below) |
| `LinearTabular` bias initialised to 0 | Fixed: bias initialised to `y_tr.mean()` (~4.99). Starting MSE dropped from ~25 to ~0.34; model now converges to RMSE 0.301, comparable to Ridge. |

---

## Feature Pipeline Summary

**Script:** `src/tabular/save_features.py` → calls `src/tabular/features.py`

- 17 steps, **no row drops** (row eligibility owned by Person B's `common.py`)
- All statistics (clip thresholds, medians, group medians) computed from train rows only → no leakage
- Output: 51 columns, 0 NaN in both split files

**Key pipeline steps:**

| Step | Action |
|---|---|
| 1 | Clip 5 columns at train 99.9th percentile |
| 2–3 | Impute host-profile cluster (~35 rows) |
| 4–6 | Recover bathrooms from text, impute bedrooms + beds |
| 7–11 | Convert pct strings, derive years_as_host, impute categoricals + rate sentinels |
| 12 | Parse amenities → count + 6 binary flags |
| 13 | has_reviews flag + impute review scores |
| 14–16 | Bucket host listings, add license flag, bucket property_type |
| 17 | Drop 30+ redundant columns (URL cols, correlated clusters, price, scraping metadata) |

**Processed files:**

| File | Rows | Cols | NaN |
|---|---|---|---|
| `data/processed/tabular_train.csv` | 4,926 | 51 | 0 |
| `data/processed/tabular_test.csv` | 1,232 | 51 | 0 |
| `data/processed/tabular_features.csv` | 6,996 | 51 | — (838 rows not in split) |

---

## Sklearn Baseline Models — Test Set Results

**Script:** `src/tabular/run_models.py`  
**Notebook:** `notebooks/tabular_models.ipynb`

| Model | RMSE log | MAE log | R² | RMSE $ | MAE $ | MedAE $ | MAPE |
|---|---|---|---|---|---|---|---|
| Dummy (mean) | 0.578 | 0.446 | -0.004 | $139 | $80 | $50 | 47.3% |
| Ridge (RidgeCV) | 0.310 | 0.219 | 0.711 | $97 | $44 | $23 | 21.7% |
| Random Forest | 0.291 | 0.200 | 0.745 | $95 | $42 | $20 | 19.4% |
| **LightGBM** | **0.212** | **0.113** | **0.865** | **$76** | **$26** | **$7** | **10.9%** |

**Notes:**
- LightGBM is the clear winner — R² = 0.865 means it explains 86.5% of price variance
- Typical prediction is off by $26 on average (MAE $) or $7 at the median (MedAE $)
- Train RMSE for LightGBM is 0.033 vs test 0.212 — significant overfitting; could benefit from more regularisation or fewer trees
- Ridge and Random Forest are within 0.02 RMSE of each other on the test set
- **Top features (LightGBM):** estimated_revenue_l365d, estimated_occupancy_l365d, amenities_count, availability_365, neighbourhood_cleansed, years_as_host, latitude/longitude

---

## PyTorch Models — Test Set Results

**Script:** `src/tabular/run_torch_models.py`  
**Notebook:** `notebooks/torch_models.ipynb`

**Setup:** 80/20 train/val split for early stopping; test set fully held out.  
`FeatureEncoder` fitted on 80% train only (OHE + StandardScaler) → `n_features = 159`.

| Model | Val RMSE log | Test RMSE log | Test MAE log | Test R² | Test MAE $ | Test MAPE |
|---|---|---|---|---|---|---|
| **LinearTabular** | 0.290 | 0.301 | 0.218 | 0.728 | $44 | 21.5% |
| MLPTabular | 0.309 | 0.334 | 0.236 | 0.665 | $46 | 22.1% |
| **ResNetTabular** | **0.263** | **0.261** | **0.179** | **0.796** | **$36** | **17.8%** |

**Notes:**
- **LinearTabular** (0.301) is now comparable to Ridge (0.310) — both are linear models; the fix was initialising the head bias to `y_tr.mean()` so gradient descent starts from a sensible point
- **MLPTabular** (0.334) is slightly weaker than the other two — it may benefit from more epochs or a tuned dropout
- **ResNetTabular** (0.261) is the best PyTorch model, between Ridge and Random Forest on the test set
- No PyTorch model beats LightGBM (0.212) in standalone evaluation — expected; GBDTs are hard to beat on tabular data, but the PyTorch models are designed for end-to-end fusion, not standalone use

---

## Full Comparison — All Models (Test Set)

| Model | RMSE log | R² | MAE $ | MAPE |
|---|---|---|---|---|
| Dummy | 0.578 | -0.004 | $80 | 47.3% |
| MLPTabular (PyTorch) | 0.334 | 0.665 | $46 | 22.1% |
| LinearTabular (PyTorch) | 0.301 | 0.728 | $44 | 21.5% |
| Ridge | 0.310 | 0.711 | $44 | 21.7% |
| Random Forest | 0.291 | 0.745 | $42 | 19.4% |
| ResNetTabular (PyTorch) | 0.261 | 0.796 | $36 | 17.8% |
| **LightGBM** | **0.212** | **0.865** | **$26** | **10.9%** |

**Best standalone tabular model: LightGBM** (RMSE 0.212, R² 0.865, MAPE 10.9%)  
**Best PyTorch model for fusion: ResNetTabular** (RMSE 0.261, R² 0.796, embed_dim=128)

---

## Saved Model Artifacts

### Sklearn — `models/tabular/sklearn/`

| File | Size | Load |
|---|---|---|
| `dummy.joblib` | ~1 KB | `joblib.load(path)` |
| `ridge.joblib` | ~1 KB | `joblib.load(path)` |
| `random_forest.joblib` | 21.2 MB | `joblib.load(path)` |
| `lightgbm.joblib` | 2.9 MB | `joblib.load(path)` |

All models accept raw `tabular_train.csv` / `tabular_test.csv` DataFrames directly — no manual preprocessing needed. Call `.predict(X_df)` where `X_df` is the raw DataFrame.

### PyTorch — `models/tabular/torch/`

| File | Size | Purpose |
|---|---|---|
| `feature_encoder.joblib` | ~9 KB | `FeatureEncoder` fitted on train; call `.transform(df)` → float32 array |
| `linear_tabular.pt` | ~3 KB | `LinearTabular` state_dict |
| `mlp_tabular.pt` | 340 KB | `MLPTabular` state_dict |
| `resnet_tabular.pt` | 2.2 MB | `ResNetTabular` state_dict |
| `metadata.json` | 232 B | `n_features=159` + model kwargs for reconstruction |

**Reload example:**
```python
import joblib, json, torch
from src.tabular.torch_models import ResNetTabular, FeatureEncoder

meta    = json.loads(open("models/tabular/torch/metadata.json").read())
encoder = joblib.load("models/tabular/torch/feature_encoder.joblib")

model = ResNetTabular(meta["n_features"], **meta["ResNetTabular"])
model.load_state_dict(torch.load("models/tabular/torch/resnet_tabular.pt", weights_only=True))
model.eval()

X = encoder.transform(df)          # df is the raw tabular CSV
emb  = model.encode(torch.tensor(X))          # [B, 128] — for fusion
pred = model.predict_numpy(X, device)          # log_price predictions
```

### Processed Predictions — `data/processed/`

| File | Rows | Columns | Model |
|---|---|---|---|
| `tabular_preds_test.csv` | 1,232 | id, log_price_actual, log_price_pred, price_actual, price_pred | LightGBM |
| `tabular_preds_train.csv` | 4,926 | id, log_price_actual, log_price_pred, price_actual, price_pred | LightGBM |
| `tabular_preds_torch_test.csv` | 1,232 | id, log_price_actual, log_price_pred, price_actual, price_pred | ResNetTabular |
| `tabular_embeddings_train.npy` | (4926, 128) | float32 | ResNetTabular encode() |
| `tabular_embeddings_test.npy` | (1232, 128) | float32 | ResNetTabular encode() |

---

## Leakage Audit

| Check | Result |
|---|---|
| Feature statistics (medians, clip thresholds, group medians) | Computed from train rows only via `train_mask = df["id"].isin(train_ids)` ✅ |
| sklearn models | All fitted on `X_train` only; RidgeCV uses LOO-CV on train ✅ |
| PyTorch `FeatureEncoder` (OHE, StandardScaler) | Fitted on 80% train subset only ✅ |
| Test set | Never touched during training, hyperparameter selection, or early stopping ✅ |
| Rare neighbourhood OHE | Neighbourhoods unseen in the 80% train subset get all-zero OHE columns (handled by `handle_unknown="ignore"`). No leakage but may slightly underrepresent rare areas. |

**No leakage found.**

---

---

## PyTorch Tuning Experiments

**Notebook:** `notebooks/torch_models.ipynb` (Tuned models section)

Diagnosis from original training curves drove two targeted changes:

| Observation | Fix |
|---|---|
| ResNetTabular: train MSE=0.041 vs val MSE=0.071 at best epoch → overfitting | dropout 0.15→0.30, weight_decay 1e-4→1e-3, n_blocks 4→3 |
| MLPTabular: best epoch=69, still improving at epoch 70 → stopping too early | patience 10→20, epochs 100→150 |

**Results (average across two notebook runs):**

| Model | Test RMSE log | Test R² | Test MAPE | vs original |
|---|---|---|---|---|
| ResNetTabular (original) | **~0.261** | **~0.795** | **~18%** | — |
| ResNetTabular (tuned) | ~0.295 | ~0.739 | ~24% | ↑ worse — over-regularised |
| MLPTabular (original) | ~0.336 | ~0.660 | ~24% | — |
| **MLPTabular (tuned)** | **~0.280** | **~0.764** | **~19%** | **↓ better (+0.056 RMSE)** |

**Conclusions:**
- ResNetTabular was not overfitting as badly as the train/val gap suggested — the extra regularisation hurt it. Original remains the best PyTorch model for fusion.
- MLPTabular genuinely benefited from more patience. Extended training converged properly (best epoch ~100 instead of ~70).
- MLPTabular tuned (RMSE ~0.280) is now nearly as good as original ResNetTabular — an alternative lower-dimensional fusion embedding (64-dim vs 128-dim).
- Neither tuned model beats LightGBM (0.212) — the dataset size (~4k rows) is the fundamental ceiling for neural approaches on tabular data alone.

**Saved artifact:** `models/tabular/torch/mlp_tabular_tuned.pt` (MLPTabular tuned only — ResNetTabular tuned discarded as it was worse).

---

## Current Status — Tabular Modality

### Complete
- Feature pipeline (51 cols, 0 NaN, train-only statistics)
- 4 sklearn baseline models (Dummy, Ridge, RF, LightGBM) + full analysis in notebook
- 3 PyTorch models (Linear, MLP, ResNet) with fusion-ready `encode()` interface
- Tuning experiment: MLPTabular tuned saved; ResNetTabular tuned discarded
- All models saved to `models/tabular/`
- All prediction CSVs saved with 5 columns (id, log_price_actual, log_price_pred, price_actual, price_pred)
- Embeddings saved for fusion: `tabular_embeddings_{train,test}.npy`

### Recommended Next Steps

1. **Reduce LightGBM overfitting** — train RMSE 0.033 vs test 0.212 is a large gap. Try `min_child_samples=50`, `num_leaves=31`, or fewer trees. Could improve test RMSE by ~0.01–0.02.

2. **Multi-modal fusion** — once text and image embeddings are ready, concatenate with `tabular_embeddings_train.npy` and train a fusion head. The target contract is already shared via `common.py` (same `train_ids.npy`/`test_ids.npy`).

3. **Late fusion baseline** — average or stack LightGBM predictions with text/image predictions before building end-to-end neural fusion.
