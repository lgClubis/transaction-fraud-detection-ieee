import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("data/raw")

    train_tx = pd.read_csv(data_dir / "train_transaction.csv")
    train_id = pd.read_csv(data_dir / "train_identity.csv")

    print("train_tx shape:", train_tx.shape)
    print("train_id shape:", train_id.shape)

    df = train_tx.merge(train_id, on="TransactionID", how="left")
    print("joined shape:", df.shape)

    print("\nisFraud distribution (counts):")
    print(df["isFraud"].value_counts(dropna=False))

    print("\nisFraud distribution (share):")
    print(df["isFraud"].value_counts(normalize=True, dropna=False))

    missing_rate = (
        df.isna()
          .mean()
          .sort_values(ascending=False)
          .head(20)
    )
    print("\nTop 20 missing-rate columns:")
    print(missing_rate)

    print("\nTransactionDT min/max:",
          df["TransactionDT"].min(),
          df["TransactionDT"].max())
    print("Unique TransactionIDs:",
          df["TransactionID"].nunique())

if __name__ == "__main__":
    main()
