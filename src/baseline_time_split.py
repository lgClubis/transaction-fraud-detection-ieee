from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    #Loading data 
    data_dir = Path("data/raw")

    tx = pd.read_csv(data_dir / "train_transaction.csv")
    ident = pd.read_csv(data_dir / "train_identity.csv")

    df = tx.merge(ident, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True) #Sort time

    #Define label
    y = df["isFraud"].astype(int)

    #Simple numeric-only baseline
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    #Remove label and id cols from features
    drop_cols = {"isFraud"}
    num_cols = [c for c in num_cols if c not in drop_cols]

    X = df[num_cols].copy()

    #Fill missing numerics with median -->efficient for baseline
    X = X.fillna(X.median(numeric_only=True))

    #Time split:
    #80% training data, 20% out-of-time validation
    n = len(df)
    split_idx = int(n * 0.8)

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Shapes:", X_train.shape, X_val.shape)
    print("Fraud rate train:", y_train.mean())
    print("Fraud rate val  :", y_val.mean())

    #Scale for logistic regression stability
    scaler = StandardScaler(with_mean=False)  #Sparse friendly even though dense
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    #Training the model
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced", #Increase costs for fraudulent transactions
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)

    val_pred = clf.predict_proba(X_val_s)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print("Validation ROC-AUC:", auc)


if __name__ == "__main__":
    main()

#Validation ROC-AUC: 0.8189410479209296
#Shapes: (472432, 402) (118108, 402)
#Fraud rate train: 0.03513521522674162
#Fraud rate val  : 0.034409184813899145