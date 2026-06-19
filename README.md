# Seattle Airbnb Price Prediction

ML course project — predicting nightly prices for Seattle Airbnb listings using the
[Inside Airbnb](http://insideairbnb.com/get-the-data/) dataset. The model is built
by combining three modalities (tabular, text, images) contributed by separate team
members and fused into a single prediction model.

## Folder structure

```
project/
├── data/
│   ├── raw/          # untouched downloads (listings.csv, calendar.csv) — not committed
│   └── processed/    # cleaned/feature outputs — not committed
├── notebooks/
│   └── tabular_explore.ipynb   # exploratory analysis for the tabular modality
├── src/
│   ├── tabular/
│   │   ├── clean.py            # data loading and price-column cleaning
│   │   ├── features.py         # numeric/categorical feature engineering
│   │   └── baseline_model.py   # baseline linear regression / random forest
│   └── shared/
│       └── split.py            # shared train/test split (see note below)
├── models/           # saved trained model artefacts
└── reports/          # figures and result tables
```

## Important: shared train/test split

**All three team members must use `src/shared/split.py` → `get_train_test_split()`
for every train/test split.**  Calling the same function with the same `random_state`
guarantees that the tabular, text, and image branches train and evaluate on exactly
the same listings, which is required for a valid fusion model.

```python
from src.shared.split import get_train_test_split

train_df, test_df = get_train_test_split(df)
```

Do not roll your own split or use a different random seed.
