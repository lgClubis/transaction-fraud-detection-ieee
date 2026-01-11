from pathlib import Path
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def main():
    data_dir= Path("data/raw")

    tx = pd.read_csv(data_dir / "train_transaction.csv")
    ident = pd.read_csv(data_dir / "train_identity.csv")

    df = tx.merge(ident, on="TransactionID", how="left")
    df = tx.sort_values("TransactionDT").reset_index(drop=True)

    y = df["isFraud"].astype(int)

    #Numerics only for fair comparison to logRegression baseline
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"isFraud", "TransactionID"}
    num_cols = [c for c in num_cols if c not in drop_cols]

    X = df[num_cols]
    n = len(df)
    split_idx = int(n * 0.8)

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Fraud rate train:", y_train.mean())
    print("Fraud rate val  :", y_val.mean())

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
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
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )
    #Checking for the 20 most important features 
    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred)
    print("Validation ROC-AUC:", auc)

    imp = pd.Series(model.feature_importance(), index=X_train.columns).sort_values(ascending=False)
    print("\nTop 20 features by importance:")
    print(imp.head(20))


    k = int(0.01 * len(y_val))  # top 1%
    top_idx = np.argsort(-val_pred)[:k]

    recall_at_1pct = y_val.iloc[top_idx].sum() / y_val.sum()
    precision_at_1pct = y_val.iloc[top_idx].mean()

    print(f"\nTop 1% review bucket (k={k}):")
    print("Recall@1%   :", recall_at_1pct)
    print("Precision@1%:", precision_at_1pct)



if __name__ == "__main__":
    main()

#Fraud rate train: 0.03513521522674162
#Fraud rate val  : 0.034409184813899145
#Training until validation scores don't improve for 50 rounds
#[50]    valid_0's auc: 0.879235
#[100]   valid_0's auc: 0.898481
#[150]   valid_0's auc: 0.905669
#[200]   valid_0's auc: 0.90857
#[250]   valid_0's auc: 0.910875
#[300]   valid_0's auc: 0.911904
#[350]   valid_0's auc: 0.912667
#[400]   valid_0's auc: 0.913501
#[450]   valid_0's auc: 0.91504
#[500]   valid_0's auc: 0.91567
#Early stopping, best iteration is:
#[490]   valid_0's auc: 0.915903
#Validation ROC-AUC: 0.915902681970103



