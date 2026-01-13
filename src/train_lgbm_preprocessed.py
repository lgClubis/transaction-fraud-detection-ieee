from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def main():
    #Using preprocessed data 
    data_path = Path("data/processed/train.parquet")
    df = pd.read_parquet(data_path)

    y = df["isFraud"].astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"isFraud", "TransactionID"}
    num_cols = [c for c in num_cols if c not in drop_cols]

    X = df[num_cols]

    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

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

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )

    val_pred = model.predict(X_val)
    k_1pct = int(0.01 * len(y_val))
    top_1pct = np.argsort(-val_pred)[:k_1pct]

    recall_1pct = y_val.iloc[top_1pct].sum() / y_val.sum()
    precision_1pct = y_val.iloc[top_1pct].mean()

    print("\nTop 1% bucket:")
    print("Recall@1%   :", recall_1pct)
    print("Precision@1%:", precision_1pct)

    imp = pd.Series(
        model.feature_importance(),
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\nTop 20 features:")
    print(imp.head(20))


    auc = roc_auc_score(y_val, val_pred)
    print("Validation ROC-AUC:", auc)

if __name__ == "__main__":
    main()