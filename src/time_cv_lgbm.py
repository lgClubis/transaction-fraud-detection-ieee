from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from encoding import target_encode
import json
from datetime import datetime


def save_timecv_run(run_name, fold_results):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = Path("runs") / f"{run_name}_timecv_{ts}.json"

    payload = {
        "run_name": run_name,
        "timestamp": ts,
        "n_folds": len(fold_results),
        "folds": fold_results,
        "summary": {
            "auc_mean": float(np.mean([f["auc"] for f in fold_results])),
            "auc_min": float(np.min([f["auc"] for f in fold_results])),
            "recall1_mean": float(np.mean([f["recall_1pct"] for f in fold_results])),
            "recall1_min": float(np.min([f["recall_1pct"] for f in fold_results])),
        }
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print("Saved Time-CV run to:", out_path)





CAT_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]


def report_bucket(y_true, scores, pct):
    k = int(pct * len(scores))
    idx= np.argsort(-scores)[:k]
    recall = y_true.iloc[idx].sum() / y_true.sum()
    precision = y_true.iloc[idx].mean()
    return recall, precision

def run_fold(df, train_end, val_end):
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()

    #Target encoding 
    for col in CAT_COLS:
        if col not in df.columns:
            continue

        tr_enc, val_enc= target_encode(
            df_train,
            df_val,
            col=col,
            target="isFraud",
            min_samples=50,
        )

        df_train[f"{col}_te"] = tr_enc.values
        df_val[f"{col}_te"] = val_enc.values

    #Features
    num_cols= df_train.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols= {"isFraud", "TransactionID"}
    num_cols= [c for c in num_cols if c not in drop_cols]

    X_train= df_train[num_cols]
    y_train= df_train["isFraud"].astype(int)

    X_val= df_val[num_cols]
    y_val= df_val["isFraud"].astype(int)

    train_data= lgb.Dataset(X_train, label=y_train)
    val_data= lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    }

    model= lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50)],
    )

    val_pred = model.predict(X_val)

    auc = roc_auc_score(y_val, val_pred)
    r1, p1 = report_bucket(y_val, val_pred, 0.01)

    return auc, r1, p1

def main():
    df = pd.read_parquet("data/processed/train.parquet")

    n = len(df)
    folds = [
        (int(0.5*n), int(0.6*n)),
        (int(0.6*n), int(0.7*n)),
        (int(0.7*n), int(0.8*n)),
        (int(0.8*n), int(0.9*n)),
    ]

    fold_results = []
    print("\n=== Time CV Results ===")
    for i, (tr_end, val_end) in enumerate(folds, 1):
        auc, r1, p1 = run_fold(df, tr_end, val_end)
        print(
            f"Fold {i}: "
            f"AUC={auc:.4f} | "
            f"Recall@1%={r1:.3f} | "
            f"Precision@1%={p1:.3f}"
        )
        fold_results.append({
        "fold": i,
        "train_end": tr_end,
        "val_end": val_end,
        "auc": float(auc),
        "recall_1pct": float(r1),
        "precision_1pct": float(p1),
    })
        
    save_timecv_run(
        run_name="lgbm_te_timecv_v1",
        fold_results=fold_results,
    )


if __name__ == "__main__":
    main()