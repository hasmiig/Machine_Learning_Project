"""
Tabular model training for the Seattle Airbnb price-prediction project.

Four models in increasing complexity:
    Dummy  →  Ridge (RidgeCV)  →  Random Forest  →  LightGBM

Every trainer accepts:
    X_train : pd.DataFrame   — raw tabular_train.csv (including 'id' column)
    y_train : pd.Series      — log_price target aligned to X_train

Every model exposes  .predict(X: pd.DataFrame) → np.ndarray  using the same
raw-DataFrame format, so the caller never needs to pre-process X manually.
"""

import numpy as np
import pandas as pd
import pandas.api.types as pat
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)
import lightgbm as lgb


# ── Column group definitions ───────────────────────────────────────────────────

# 't'/'f' string columns converted to 0/1 before any model
_BOOL_STR = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
]

# Rate columns that use -1 as a "no data" sentinel
_RATE_COLS = ["host_response_rate", "host_acceptance_rate"]

# Ordered categories for host_response_time (best → worst → unknown)
_RESPONSE_TIME_ORDER = [
    "within an hour",
    "within a few hours",
    "within a day",
    "a few days or more",
    "unknown",
]

# Low-cardinality nominal string columns (≤ 9 unique values)
_OHE_COLS = [
    "host_verifications",
    "room_type",
    "property_type",
    "host_listing_bucket",
]

# High-cardinality neighbourhood (87 unique) — target-encoded for Ridge,
# handled natively for tree models
_NEIGHBOURHOOD = "neighbourhood_cleansed"

# Columns dropped at model time (not predictive / redundant with _NEIGHBOURHOOD)
_DROP_COLS = ["id", "neighbourhood_group_cleansed"]

# All string categoricals present after _prepare_common
_CAT_STR_COLS = [
    "host_response_time",
    "host_verifications",
    "room_type",
    "property_type",
    "host_listing_bucket",
    "neighbourhood_cleansed",
]


# ── Shared pre-processing ──────────────────────────────────────────────────────

def _prepare_common(X: pd.DataFrame) -> pd.DataFrame:
    """Drop id + redundant neighbourhood; convert bool strings to 0/1 int."""
    X = X.copy()
    X = X.drop(columns=[c for c in _DROP_COLS if c in X.columns])
    for col in _BOOL_STR:
        if col in X.columns:
            X[col] = (X[col] == "t").astype(np.int8)
    return X


def _prepare_for_ridge(X: pd.DataFrame) -> pd.DataFrame:
    """Common prep + replace -1 rate sentinels with NaN (imputed in pipeline)."""
    X = _prepare_common(X)
    for col in _RATE_COLS:
        if col in X.columns:
            X[col] = X[col].replace(-1.0, np.nan)
    return X


def _prepare_for_tree(X: pd.DataFrame) -> pd.DataFrame:
    """-1 sentinels kept; trees split on them directly as a 'no-data' signal."""
    return _prepare_common(X)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    y_true: pd.Series,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """Print and return RMSE in log-price space and dollar space."""
    rmse_log = float(np.sqrt(np.mean((y_true.values - y_pred) ** 2)))
    rmse_dollar = float(np.sqrt(np.mean((np.exp(y_true.values) - np.exp(y_pred)) ** 2)))
    prefix = f"[{label}]" if label else ""
    print(f"{prefix:<22}  RMSE log={rmse_log:.4f}   RMSE $={rmse_dollar:>10,.0f}")
    return {"rmse_log": rmse_log, "rmse_dollar": rmse_dollar}


# ── 1. Dummy ───────────────────────────────────────────────────────────────────

def train_dummy(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Predict the training-set mean log_price for every row — absolute floor."""
    pipe = Pipeline([
        ("prep",  FunctionTransformer(_prepare_for_tree, validate=False)),
        ("model", DummyRegressor(strategy="mean")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ── 2. Ridge ───────────────────────────────────────────────────────────────────

def train_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    RidgeCV with full sklearn preprocessing:
    - StandardScaler           on all numeric cols
    - Median imputation + scale on rate cols (after -1 → NaN conversion)
    - OrdinalEncoder + scale    on host_response_time
    - OneHotEncoder             on low-cardinality nominal cols (≤ 9 unique)
    - TargetEncoder             on neighbourhood_cleansed (87 unique)
    Alpha selected via leave-one-out CV over [0.1, 1, 10, 100, 1000, 10000].
    """
    # Compute column groups from a prepared sample of the training data
    X_tr = _prepare_for_ridge(X_train)

    num_cols = [
        c for c in X_tr.columns
        if c not in _RATE_COLS
        and c not in _CAT_STR_COLS
        and pat.is_numeric_dtype(X_tr[c])
    ]
    rate_present = [c for c in _RATE_COLS     if c in X_tr.columns]
    ohe_present  = [c for c in _OHE_COLS      if c in X_tr.columns]
    nb_present   = [_NEIGHBOURHOOD]            if _NEIGHBOURHOOD in X_tr.columns else []
    ord_present  = ["host_response_time"]      if "host_response_time" in X_tr.columns else []

    ct = ColumnTransformer(
        transformers=[
            ("num",
             StandardScaler(),
             num_cols),
            ("rate",
             Pipeline([
                 ("imp", SimpleImputer(strategy="median")),
                 ("scl", StandardScaler()),
             ]),
             rate_present),
            ("ordinal",
             Pipeline([
                 ("oe", OrdinalEncoder(
                     categories=[_RESPONSE_TIME_ORDER],
                     handle_unknown="use_encoded_value",
                     unknown_value=len(_RESPONSE_TIME_ORDER),
                 )),
                 ("scl", StandardScaler()),
             ]),
             ord_present),
            ("ohe",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ohe_present),
            ("te",
             TargetEncoder(target_type="continuous"),
             nb_present),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("prep",  FunctionTransformer(_prepare_for_ridge, validate=False)),
        ("ct",    ct),
        ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0])),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ── 3. Random Forest ───────────────────────────────────────────────────────────

def train_rf(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    RandomForestRegressor with OrdinalEncoder on all string categoricals.
    -1 rate sentinels are kept — tree splits on them signal 'no host response data'.
    """
    X_tr = _prepare_for_tree(X_train)

    cat_present = [c for c in _CAT_STR_COLS if c in X_tr.columns]
    num_cols    = [c for c in X_tr.columns  if c not in cat_present]

    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat",
             OrdinalEncoder(
                 handle_unknown="use_encoded_value",
                 unknown_value=-1,
             ),
             cat_present),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("prep",  FunctionTransformer(_prepare_for_tree, validate=False)),
        ("ct",    ct),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ── 4. LightGBM ────────────────────────────────────────────────────────────────

class LGBMTabularModel:
    """
    LightGBM wrapper with native categorical support.

    String categoricals are converted to pandas Categorical dtype on both train
    and test, preserving the training category vocabulary so test rows with
    unseen categories are treated as NaN (LGBM's built-in missing-value path).
    """

    def __init__(self) -> None:
        self._model: lgb.LGBMRegressor | None = None
        self._cat_cols: list[str] = []
        self._cat_dtypes: dict = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LGBMTabularModel":
        X = _prepare_for_tree(X_train)
        self._cat_cols = [c for c in _CAT_STR_COLS if c in X.columns]
        for col in self._cat_cols:
            cat = pd.Categorical(X[col])
            self._cat_dtypes[col] = cat.dtype
            X[col] = cat

        self._model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self._model.fit(X, y_train, categorical_feature="auto")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = _prepare_for_tree(X)
        for col in self._cat_cols:
            if col in X.columns:
                X[col] = pd.Categorical(
                    X[col], categories=self._cat_dtypes[col].categories
                )
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

    @property
    def feature_name_(self) -> list[str]:
        return self._model.feature_name_


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMTabularModel:
    """LightGBM with 500 trees, native categorical support, and L2 regularisation."""
    return LGBMTabularModel().fit(X_train, y_train)
