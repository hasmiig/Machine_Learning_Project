# Multimodal Airbnb Price Prediction — Seattle

ML course project (Summer Semester 2026) predicting nightly Airbnb prices in Seattle from
three data modalities: tabular listing attributes, text (descriptions + reviews), and listing
photos. Each modality was built independently by one team member against a shared
train/test contract, then combined into a **fused multimodal model**, which is the central
deliverable of the project.

**Team:** Hasmig Aintablian, Paula Bustos, Ali Hessam.

## Research question

Does multimodal learning improve price-prediction performance over any individual modality
alone?

**Answer:** Yes, modestly. Fusing tabular + text (R² = 0.753) beats the best single modality
(tabular alone, R² ≈ 0.74–0.75) by roughly +0.01 R². Image does not add further value on top
and can mildly hurt naive fusion — but a stacked (late-fusion) model that can down-weight
image reaches the same ceiling without that risk.

## Dataset

- Source: [Inside Airbnb](http://insideairbnb.com/get-the-data/), Seattle snapshot scraped
  2025-09-25.
- 6,996 raw listings × 79 columns. Target: `log_price = log(price)` (raw price $15–$50,039,
  median $149, heavily right-skewed).
- Fixed 80/20 train/test split (4,926 train / 1,232 test), shared by **all three modalities**
  so every model is scored on exactly the same listings. 775 listings (11.1%) with no price
  are excluded from the split entirely.
- The target and split are built once by `src/shared/common.py::build_target_and_split()` and
  reloaded everywhere via `load_target()` / `load_split()` — no modality should roll its own
  split.

## Repository structure

```
project/
├── data/
│   ├── raw/            # listings.csv, calendar.csv, neighbourhoods.csv, target.csv,
│   │                    # train_ids.npy / test_ids.npy (not committed)
│   └── processed/       # tabular_features/train/test.csv — save_features.py output (not committed)
├── src/
│   ├── shared/
│   │   ├── common.py               # target + fixed train/test split, used by every modality
│   │   ├── prepare_features.ipynb  # builds the shared multimodal feature bundle
│   │   └── model_training.ipynb    # THE MAIN MODEL: early-fusion grid + late-fusion stacking
│   ├── tabular/
│   │   ├── clean.py, features.py, save_features.py    # cleaning + 17-step feature pipeline
│   │   └── tabular_explore.ipynb                       # EDA only — no standalone tabular models here
│   ├── text/
│   │   └── text_features.ipynb     # TF-IDF + sentence-embedding pipeline
│   └── images/
│       ├── Images_availability.ipynb  # photo-availability check
│       ├── Images_ResNet2.ipynb       # ResNet50 regressor + embedding extraction
│       └── Grad_cam.ipynb             # Grad-CAM interpretability check
```

Note: earlier drafts of this project trained standalone per-modality models (a tabular-only
sklearn/PyTorch pipeline, separate from fusion) to sanity-check that each modality carried real
signal before fusing. Those standalone tabular model scripts/notebooks have since been removed —
they were exploratory scaffolding, not the deliverable. The single-modality numbers you still see
below (e.g. "tabular alone") now come directly from `model_training.ipynb`, which tests every
feature set — including each modality alone — as part of the same fusion grid.

## Pipeline overview

```
Shared Airbnb listings
   ├─ Tabular features  (amenities, host info, location, availability, ...)
   ├─ Text features      (name + description + reviews → TF-IDF / sentence embeddings)
   └─ Image features     (listing photos → ResNet50 embeddings)
              │
              ▼
   Feature-level fusion (concatenation): 7 feature sets
   {tabular, text, image, tab+text, tab+img, text+img, tab+text+img}
              │
              ▼
   7 models per feature set: Dummy, Ridge, kNN, Random Forest, XGBoost, LightGBM, MLP
              │
              ▼
   Predict log(price) → evaluate on the fixed, shared test split
   (R², RMSE, MAE, MedAE, MAPE, Explained Variance)

   + separately, prediction-level fusion (stacking): one model per modality produces
     out-of-fold predictions → a Ridge meta-model combines them into a final prediction
```

## Per-modality preprocessing

**Tabular** (`src/tabular/features.py`) — a 17-step pipeline: recover/impute bathrooms,
bedrooms, host-profile fields; parse amenities into a count + 6 binary flags; derive
`years_as_host`; bucket host-listing counts and property types; drop 30+ redundant, correlated,
or URL/metadata columns. All statistics (clip thresholds, medians, category frequencies) are
fit on train rows only. Output: `tabular_train.csv` / `tabular_test.csv` (49 features, 0 NaNs) —
still contains raw category columns (`room_type`, `neighbourhood_cleansed`, ...), which
`prepare_features.ipynb` (below) converts to pure numbers for fusion. **Note:**
`estimated_revenue_l365d` — Inside Airbnb's own price × occupancy column — is deliberately
dropped as a target leak; keeping it inflates tabular R² from 0.745 to 0.871 in a controlled
comparison (see `prepare_features.ipynb`).

**Text** (`src/text/text_features.ipynb`) — one row per listing: `name + description`, plus
reviews aggregated per listing (most recent 30, never crossing the train/test boundary; 722
listings with no reviews get a zero vector + `has_reviews` flag rather than being dropped). Two
feature sets: TF-IDF (5,000-word vocabulary) and `all-MiniLM-L6-v2` sentence embeddings
(770 dims: name+description + averaged review embeddings).

**Image** (`src/images/`) — photos are downloadable for ~94% of listings; missingness is
unrelated to price (Mann-Whitney p = 0.92), so it's safely imputed. An ImageNet-pretrained
ResNet50 (frozen except the last residual block) is fine-tuned as a price regressor; its
2048-dim penultimate-layer embeddings are kept regardless of the regression head's own
performance, with a mean-vector fallback for missing photos.

**Fusion feature bundle** (`src/shared/prepare_features.ipynb`) — independently re-encodes all
three modalities onto the shared 6,158-listing spine (one-hot instead of target-encoding for
`neighbourhood_cleansed`, to avoid leaking the target into a matrix shared by every model):
tabular → 152 dims, text → 100 dims (TruncatedSVD, 78.8% variance retained), image → 50 dims
(TruncatedSVD, 69.2% variance retained). Saved as `feature_bundle.npz`.

## Results

### Single-modality strength (from the fusion grid, one modality at a time)

| Modality | Best model | Test R² |
|---|---|---|
| Tabular | XGBoost | 0.742 |
| Text | MLP | 0.608 |
| Image | Random Forest / XGBoost | 0.250 |
| Dummy | — | -0.004 |

These come straight out of `model_training.ipynb`'s grid (the "tabular", "text", "image" rows of
the table below) — there's no separate standalone-model codebase behind them anymore.


### Early fusion — feature concatenation (test R²)

| Feature set | Ridge | RF | XGBoost | LightGBM | MLP |
|---|---|---|---|---|---|
| tabular | 0.672 | 0.714 | 0.742 | 0.738 | 0.716 |
| text | 0.547 | 0.530 | 0.578 | 0.590 | 0.608 |
| image | 0.180 | 0.250 | 0.250 | 0.231 | 0.180 |
| tabular + text | 0.705 | 0.708 | 0.752 | **0.753** | 0.731 |
| tabular + image | 0.678 | 0.702 | 0.735 | 0.732 | 0.711 |
| text + image | 0.551 | 0.516 | 0.592 | 0.591 | 0.572 |
| tabular + text + image | 0.706 | 0.704 | 0.750 | 0.751 | 0.713 |

**Best: tabular + text, LightGBM, R² = 0.753.** Adding image on top of tabular+text makes
results slightly *worse*, not better — naive concatenation gets dragged down by the
weaker/noisier image signal.

### Late fusion — prediction-level stacking

Per-modality XGBoost models produce out-of-fold train predictions; a RidgeCV meta-model
combines them:

| Base model | Test R² |
|---|---|
| tabular | 0.742 |
| text | 0.578 |
| image | 0.250 |
| **Stacked (Ridge meta)** | **0.753** |

Learned weights: tabular +0.857, text +0.177, image +0.083 — the meta-model rediscovers the
tabular > text > image ordering without manual tuning, ties the best early-fusion result, and
explicitly down-weights the noisy image signal instead of being dragged down by it.

### Verdict

Multimodal fusion **does** beat the best single modality, though modestly (+0.01 R²) and
specifically because of text. Two independent fusion strategies (concatenation and stacking)
converge on the same R² ≈ 0.753 ceiling, which is good evidence the result is robust rather
than an artifact of one method.

## Key findings

- **Modality strength tracks how directly the data encodes price.** Structured
  location/amenity/host data is closest to what drives Seattle nightly rates; text adds real
  secondary signal; photos alone barely help a plain regression head.
- **Embedding quality ≠ end-to-end prediction accuracy.** The image regressor failed as a
  standalone predictor (R² = -0.288) even though its intermediate embeddings carried usable
  signal for other models.
- **Naive concatenation is sensitive to modality imbalance;** stacking's explicit per-modality
  weighting avoids the small degradation concatenation suffers when image is added.

## Reproducibility

The tabular *feature* pipeline runs locally end-to-end from the project root:

```bash
python src/tabular/save_features.py
```

This produces `tabular_features.csv` / `tabular_train.csv` / `tabular_test.csv` in
`data/processed/`. Re-run `tabular_explore.ipynb` afterwards to regenerate the EDA notebook
against them. No models are trained locally at this stage — this step only produces clean,
engineered features.

All actual modeling — the single-modality baselines, the 7×7 fusion grid, and the stacking
model — lives in `src/shared/prepare_features.ipynb` and `src/shared/model_training.ipynb`.
These were developed in Google Colab against a shared Google Drive folder — `common.py`'s
`DATA_DIR` points at that Drive path, as do the text (`src/text/text_features.ipynb`) and image
(`src/images/*.ipynb`) notebooks. To run any of them locally, point `DATA_DIR` at `data/raw/`
instead; otherwise, their baked-in notebook outputs are the source of truth for the results
reported above.

