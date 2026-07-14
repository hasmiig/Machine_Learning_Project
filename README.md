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
│   └── processed/       # engineered tabular features, predictions, embeddings (not committed)
├── src/
│   ├── shared/
│   │   ├── common.py               # target + fixed train/test split, used by every modality
│   │   ├── prepare_features.ipynb  # builds the shared multimodal feature bundle
│   │   └── model_training.ipynb    # FUSION: early-fusion grid + late-fusion stacking
│   ├── tabular/
│   │   ├── clean.py, features.py, save_features.py    # cleaning + feature pipeline
│   │   ├── model.py, run_models.py                     # Dummy / Ridge / Random Forest / LightGBM
│   │   ├── torch_models.py, run_torch_models.py        # Linear / MLP / ResNet, expose .encode()
│   │   ├── tabular_explore.ipynb, tabular_models.ipynb, torch_models.ipynb
│   │   └── models/{sklearn,torch}/                     # saved model artifacts
│   ├── text/
│   │   └── text_features.ipynb     # TF-IDF + sentence-embedding pipeline
│   └── images/
│       ├── Images_availability.ipynb  # photo-availability check
│       ├── Images_ResNet2.ipynb       # ResNet50 regressor + embedding extraction
│       └── Grad_cam.ipynb             # Grad-CAM interpretability check
└── reports/             # earlier process reports; superseded by this README
```

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
fit on train rows only. **Note:** `estimated_revenue_l365d` — Inside Airbnb's own
price × occupancy column — is deliberately dropped as a target leak; keeping it inflates
LightGBM R² from 0.746 to 0.871.

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

### Standalone modality strength (best model per modality)

| Modality | Best model | Test R² |
|---|---|---|
| Tabular | LightGBM | 0.746 (own pipeline) / 0.742 (fusion re-encoding) |
| Text | MLP | 0.608 |
| Image | Random Forest / XGBoost | 0.250 |
| Dummy | — | -0.004 |


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

The tabular pipeline runs locally end-to-end from the project root:

```bash
python src/tabular/save_features.py
python src/tabular/run_models.py
python src/tabular/run_torch_models.py
```

then re-run `tabular_explore.ipynb`, `tabular_models.ipynb`, and `torch_models.ipynb` to
regenerate the analysis notebooks.

The text, image, and fusion notebooks (`src/text/text_features.ipynb`, `src/images/*.ipynb`,
`src/shared/prepare_features.ipynb`, `src/shared/model_training.ipynb`) were developed in
Google Colab against a shared Google Drive folder — `common.py`'s `DATA_DIR` points at that
Drive path. To run them locally, point `DATA_DIR` at `data/raw/` instead; otherwise, their
baked-in notebook outputs are the source of truth for the results reported above.

