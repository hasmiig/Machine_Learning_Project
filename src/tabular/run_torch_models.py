"""
Train and evaluate PyTorch tabular models for Seattle Airbnb price prediction.

Usage (from project root):
    python src/tabular/run_torch_models.py

Outputs written to data/processed/:
    tabular_preds_torch_test.csv     — ResNetTabular test-set predictions (id, log_price_actual, log_price_pred, price_actual, price_pred)
    tabular_embeddings_train.npy     — float32 array [N_train, embed_dim]
    tabular_embeddings_test.npy      — float32 array [N_test,  embed_dim]

Outputs written to models/tabular/torch/:
    feature_encoder.joblib           — fitted FeatureEncoder (for inference)
    linear_tabular.pt                — LinearTabular state_dict
    mlp_tabular.pt                   — MLPTabular state_dict
    resnet_tabular.pt                — ResNetTabular state_dict
    metadata.json                    — n_features + model kwargs for reconstruction
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.tabular.torch_models import (  # noqa: E402
    FeatureEncoder,
    LinearTabular,
    MLPTabular,
    ResNetTabular,
    _TabularBase,
    evaluate_model,
    train_model,
)

_DATA_PROCESSED = _ROOT / "data" / "processed"
_DATA_RAW = _ROOT / "data" / "raw"


def _get_embeddings(
    model: _TabularBase,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(
                X[start : start + batch_size], dtype=torch.float32, device=device
            )
            parts.append(model.encode(batch).cpu().numpy())
    return np.vstack(parts)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load ───────────────────────────────────────────────────────────────────
    df_train_raw = pd.read_csv(_DATA_PROCESSED / "tabular_train.csv")
    df_test_raw  = pd.read_csv(_DATA_PROCESSED / "tabular_test.csv")
    target       = pd.read_csv(_DATA_RAW / "target.csv").set_index("id")

    train_ids = df_train_raw["id"].values
    test_ids  = df_test_raw["id"].values

    y_train_full = target.loc[train_ids, "log_price"].values.astype(np.float32)
    y_test       = target.loc[test_ids,  "log_price"].values.astype(np.float32)

    # ── 80 / 20 train / val split for early stopping ──────────────────────────
    idx = np.arange(len(df_train_raw))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_tr_df  = df_train_raw.iloc[tr_idx].reset_index(drop=True)
    X_val_df = df_train_raw.iloc[val_idx].reset_index(drop=True)
    y_tr  = y_train_full[tr_idx]
    y_val = y_train_full[val_idx]

    # ── Fit encoder on X_train only — no leakage from val or test ─────────────
    encoder = FeatureEncoder()
    X_tr   = encoder.fit_transform(X_tr_df)
    X_val  = encoder.transform(X_val_df)
    X_test = encoder.transform(df_test_raw)

    n = encoder.n_features
    print(f"Encoded feature dimension: {n}\n")

    # ── Models ─────────────────────────────────────────────────────────────────
    # LinearTabular bias initialised to y_train mean so gradient descent starts
    # from the dummy baseline (MSE ≈ 0.34) not zero (MSE ≈ 25).
    y_mean = float(y_tr.mean())
    models: dict[str, _TabularBase] = {
        "LinearTabular": LinearTabular(n, bias_init=y_mean),
        "MLPTabular":    MLPTabular(n),
        "ResNetTabular": ResNetTabular(n),
    }

    results: dict[str, dict] = {}

    for name, model in models.items():
        print("=" * 55)
        print(f"  {name}")
        print("=" * 55)

        train_model(
            model, X_tr, y_tr, X_val, y_val,
            epochs=100,
            batch_size=256,
            lr=1e-3,
            weight_decay=1e-4,
            patience=10,
            device=device,
            verbose=True,
        )

        print()
        val_m  = evaluate_model(model, X_val,  y_val,  f"{name} val",  device)
        test_m = evaluate_model(model, X_test, y_test, f"{name} test", device)
        results[name] = {"val": val_m, "test": test_m}
        print()

    # ── Summary table ──────────────────────────────────────────────────────────
    sep = "=" * 95
    print(f"\n{sep}")
    print(f"{'Model':<20}  {'Val RMSE log':>12}  {'Test RMSE log':>13}  {'Test MAE log':>12}  "
          f"{'Test R²':>7}  {'Test MAE $':>10}  {'Test MAPE%':>10}")
    print("-" * 95)
    for name, r in results.items():
        v, t = r["val"], r["test"]
        print(
            f"{name:<20}  "
            f"{v['rmse_log']:>12.4f}  "
            f"{t['rmse_log']:>13.4f}  "
            f"{t['mae_log']:>12.4f}  "
            f"{t['r2']:>7.4f}  "
            f"{t['mae_dollar']:>10,.0f}  "
            f"{t['mape']:>10.1f}"
        )
    print(sep)

    # ── Save ResNetTabular test predictions (log, dollar, and actual price) ───
    resnet = models["ResNetTabular"]
    test_preds = resnet.predict_numpy(X_test, device)
    pred_path = _DATA_PROCESSED / "tabular_preds_torch_test.csv"
    pd.DataFrame({
        "id":               test_ids,
        "log_price_actual": y_test,
        "log_price_pred":   test_preds,
        "price_actual":     np.exp(y_test),
        "price_pred":       np.exp(test_preds),
    }).to_csv(pred_path, index=False)
    print(f"\nSaved predictions  → {pred_path}  [id, log_price_actual, log_price_pred, price_actual, price_pred]")

    # ── Save ResNetTabular embeddings (full train + test) for fusion ───────────
    # Encode the entire tabular_train.csv (all 4926 rows), not just the 80 % split,
    # so downstream fusion has an embedding for every training sample.
    X_train_full = encoder.transform(df_train_raw)
    train_emb = _get_embeddings(resnet, X_train_full, device)
    test_emb  = _get_embeddings(resnet, X_test,       device)

    train_emb_path = _DATA_PROCESSED / "tabular_embeddings_train.npy"
    test_emb_path  = _DATA_PROCESSED / "tabular_embeddings_test.npy"
    np.save(train_emb_path, train_emb)
    np.save(test_emb_path,  test_emb)

    print(
        f"Saved embeddings   → train {train_emb.shape}  test {test_emb.shape}\n"
        f"                     {train_emb_path}\n"
        f"                     {test_emb_path}"
    )

    # ── Save models and encoder ────────────────────────────────────────────────
    torch_dir = _ROOT / "models" / "tabular" / "torch"
    torch_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoder, torch_dir / "feature_encoder.joblib")
    torch.save(models["LinearTabular"].state_dict(),  torch_dir / "linear_tabular.pt")
    torch.save(models["MLPTabular"].state_dict(),      torch_dir / "mlp_tabular.pt")
    torch.save(models["ResNetTabular"].state_dict(),   torch_dir / "resnet_tabular.pt")

    # Metadata needed to reconstruct models at inference time
    metadata = {
        "n_features": n,
        "LinearTabular":  {},
        "MLPTabular":     {"hidden_dims": [256, 128, 64], "dropout": 0.2},
        "ResNetTabular":  {"d_model": 128, "n_blocks": 4, "dropout": 0.15},
    }
    (torch_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved PyTorch models → {torch_dir}")
    for f in sorted(torch_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name}  ({size / 1e6:.2f} MB)" if size > 1e5 else f"  {f.name}  ({size} B)")


if __name__ == "__main__":
    main()
