from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import json
from datetime import datetime
from encoding import target_encode


def save_run(run_name, metrics):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = Path("runs") / f"{run_name}_{ts}.json"

    payload = {
        "run_name": run_name,
        "timestamp": ts,
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("Saved run to:", out_path)

def report_bucket(y_true, scores, pct):
    k = int(pct*len(scores))
    idX= np.argsort(-scores)[:k]
    recall = y_true.iloc[idX].sum() / y_true.sum()
    precision = y_true.iloc[idX].mean()
    return recall, precision

def main():
    data_path = Path("data/processed/train.parquet")
    model_path = Path("models/lgbm_te_v1.txt")

    print("Loading data...")
    df = pd.read_parquet(data_path)

    y = df["isFraud"].astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"isFraud", "TransactionID"}
    num_cols = [c for c in num_cols if c not in drop_cols]

    CAT_COLS = [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        ]

    split_idx = int(len(df) * 0.8)

    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()

    # Target enxoding like in the model training
    for col in CAT_COLS:
        if col not in df.columns:
            continue

        tr_enc, val_enc = target_encode(
            df_train,
            df_val,
            col=col,
            target="isFraud",
            min_samples=50,
        )

        df_train[f"{col}_te"] = tr_enc.values
        df_val[f"{col}_te"] = val_enc.values

    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"isFraud", "TransactionID"}
    num_cols = [c for c in num_cols if c not in drop_cols]

    X_val = df_val[num_cols]
    y_val = df_val["isFraud"].astype(int)


    print("Loading model...")
    model = lgb.Booster(model_file=str(model_path))
    expected = model.feature_name()
    print("Model expects", len(expected), "features")

    #Train and evaluate expected different amounts of features
    expected = model.feature_name()
    print("Model expects", len(expected), "features")

    X_val_full = X_val.copy()

    #Adding missing features
    missing = [c for c in expected if c not in X_val_full.columns]
    for c in missing:
        X_val_full[c] = np.nan

    #Remove features not needed in correct order
    X_val = X_val_full[expected]

    print("Eval features:", X_val.shape[1])




    print("Scoring validation set...")
    val_pred = model.predict(X_val)

    auc = roc_auc_score(y_val, val_pred)
    print("\nValidation ROC-AUC:", auc)

    print("\n=== Review Buckets ===")
    for pct in [0.005, 0.01, 0.05]:
        r, p = report_bucket(y_val, val_pred, pct)
        print(f"Top {pct*100:.1f}% bucket:")
        print(f"  Recall   : {r:.3f}")
        print(f"  Precision: {p:.3f}")

    bucket_results = []
    for pct in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]:
        r, p = report_bucket(y_val, val_pred, pct)
        k = int(pct * len(y_val))
        print(f"Top {pct*100:>4.1f}% (k={k:>6}): Recall={r:.3f}  Precision={p:.3f}")
        bucket_results.append({
            "pct": pct,
            "k": k,
            "recall": float(r),
            "precision": float(p),
        })
    metrics = {
    "roc_auc": float(auc),
    "bucket_curve": bucket_results,
    }

    save_run("lgbm_te_v1", metrics)    


if __name__ == "__main__":
    main()
