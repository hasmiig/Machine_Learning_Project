"""
Train and evaluate all four tabular models on the Seattle Airbnb dataset.

Output
------
  Console  — per-model RMSE (log-price and dollar), summary table, best alpha
  data/processed/tabular_preds_test.csv      — id, log_price_actual, log_price_pred, price_actual, price_pred  (LightGBM)
  data/processed/tabular_preds_train.csv     — id, log_price_actual, log_price_pred, price_actual, price_pred  (LightGBM)
  models/tabular/sklearn/dummy.joblib
  models/tabular/sklearn/ridge.joblib
  models/tabular/sklearn/random_forest.joblib
  models/tabular/sklearn/lightgbm.joblib

Run from the project root:
    python src/tabular/run_models.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.tabular.model import (  # noqa: E402
    evaluate,
    train_dummy,
    train_lgbm,
    train_rf,
    train_ridge,
)


def load_data():
    X_train = pd.read_csv(_ROOT / "data/processed/tabular_train.csv")
    X_test  = pd.read_csv(_ROOT / "data/processed/tabular_test.csv")
    target  = pd.read_csv(_ROOT / "data/raw/target.csv")

    tgt     = target.set_index("id")["log_price"]
    y_train = X_train["id"].map(tgt)
    y_test  = X_test["id"].map(tgt)

    # Drop any rows with missing target (shouldn't occur with valid split files)
    mask_tr = y_train.notna()
    mask_te = y_test.notna()
    if not mask_tr.all() or not mask_te.all():
        print(
            f"Warning: {(~mask_tr).sum()} train / {(~mask_te).sum()} test rows "
            "have no target — dropping them."
        )
        X_train = X_train[mask_tr].reset_index(drop=True)
        y_train = y_train[mask_tr].reset_index(drop=True)
        X_test  = X_test[mask_te].reset_index(drop=True)
        y_test  = y_test[mask_te].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def main():
    print("Loading data …")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {len(X_train):,} rows   Test: {len(X_test):,} rows\n")

    results = []

    # ── 1. Dummy ───────────────────────────────────────────────────────────────
    print("── 1. Dummy Regressor " + "─" * 50)
    dummy  = train_dummy(X_train, y_train)
    r_tr   = evaluate(y_train, dummy.predict(X_train), "dummy  train")
    r_te   = evaluate(y_test,  dummy.predict(X_test),  "dummy  test ")
    results.append({"model": "Dummy",        **_prefix(r_tr, "train"), **_prefix(r_te, "test")})

    # ── 2. Ridge ───────────────────────────────────────────────────────────────
    print("\n── 2. Ridge (RidgeCV) " + "─" * 50)
    ridge  = train_ridge(X_train, y_train)
    print(f"  Best alpha: {ridge['model'].alpha_:.1f}")
    r_tr   = evaluate(y_train, ridge.predict(X_train), "ridge  train")
    r_te   = evaluate(y_test,  ridge.predict(X_test),  "ridge  test ")
    results.append({"model": "Ridge",        **_prefix(r_tr, "train"), **_prefix(r_te, "test")})

    # ── 3. Random Forest ───────────────────────────────────────────────────────
    print("\n── 3. Random Forest (300 trees) " + "─" * 40)
    rf     = train_rf(X_train, y_train)
    r_tr   = evaluate(y_train, rf.predict(X_train), "rf     train")
    r_te   = evaluate(y_test,  rf.predict(X_test),  "rf     test ")
    results.append({"model": "RandomForest", **_prefix(r_tr, "train"), **_prefix(r_te, "test")})

    # ── 4. LightGBM ────────────────────────────────────────────────────────────
    print("\n── 4. LightGBM (500 trees) " + "─" * 44)
    lgbm   = train_lgbm(X_train, y_train)
    r_tr   = evaluate(y_train, lgbm.predict(X_train), "lgbm   train")
    r_te   = evaluate(y_test,  lgbm.predict(X_test),  "lgbm   test ")
    results.append({"model": "LightGBM",    **_prefix(r_tr, "train"), **_prefix(r_te, "test")})

    # ── Summary table ──────────────────────────────────────────────────────────
    summary = (
        pd.DataFrame(results)
        .set_index("model")
        .rename(columns={
            "test_rmse_log":         "RMSE (log)",
            "test_mae_log":          "MAE (log)",
            "test_r2":               "R²",
            "test_rmse_dollar":      "RMSE ($)",
            "test_mae_dollar":       "MAE ($)",
            "test_median_ae_dollar": "MedAE ($)",
            "test_mape":             "MAPE (%)",
        })
    )
    test_cols = ["RMSE (log)", "MAE (log)", "R²", "RMSE ($)", "MAE ($)", "MedAE ($)", "MAPE (%)"]
    print("\n\n── Test-set summary " + "─" * 52)
    print(summary[test_cols].to_string(
        float_format=lambda x: f"{x:,.4f}" if abs(x) < 10 else f"{x:,.1f}" if abs(x) < 1000 else f"{x:,.0f}"
    ))

    # ── Save LightGBM predictions (log, dollar, and actual price) ─────────────
    out_dir = _ROOT / "data/processed"
    test_log_pred  = lgbm.predict(X_test)
    train_log_pred = lgbm.predict(X_train)

    pd.DataFrame({
        "id":              X_test["id"].values,
        "log_price_actual": y_test.values,
        "log_price_pred":   test_log_pred,
        "price_actual":     np.exp(y_test.values),
        "price_pred":       np.exp(test_log_pred),
    }).to_csv(out_dir / "tabular_preds_test.csv", index=False)

    pd.DataFrame({
        "id":              X_train["id"].values,
        "log_price_actual": y_train.values,
        "log_price_pred":   train_log_pred,
        "price_actual":     np.exp(y_train.values),
        "price_pred":       np.exp(train_log_pred),
    }).to_csv(out_dir / "tabular_preds_train.csv", index=False)

    print(f"\nSaved  tabular_preds_test.csv   ({len(X_test):,} rows)  [id, log_price_actual, log_price_pred, price_actual, price_pred]")
    print(f"Saved  tabular_preds_train.csv  ({len(X_train):,} rows)  [id, log_price_actual, log_price_pred, price_actual, price_pred]")

    # ── Save all sklearn models ────────────────────────────────────────────────
    models_dir = _ROOT / "models" / "tabular" / "sklearn"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dummy, models_dir / "dummy.joblib")
    joblib.dump(ridge, models_dir / "ridge.joblib")
    joblib.dump(rf,    models_dir / "random_forest.joblib")
    joblib.dump(lgbm,  models_dir / "lightgbm.joblib")
    print(f"\nSaved sklearn models → {models_dir}")
    for f in sorted(models_dir.glob("*.joblib")):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

    # ── Top-20 LightGBM feature importances ───────────────────────────────────
    imp = pd.Series(
        lgbm.feature_importances_,
        index=lgbm.feature_name_,
    ).sort_values(ascending=False)
    print("\nTop 20 LightGBM features (split gain):")
    print(imp.head(20).to_string())


def _prefix(d: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


if __name__ == "__main__":
    main()
