from pathlib import Path
import pandas as pd

def main():
    raw_dir= Path("data/raw")
    out_dir= Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    tx= pd.read_csv(raw_dir / "train_transaction.csv")
    ident= pd.read_csv(raw_dir / "train_identity.csv")

    df= tx.merge(ident, on="TransactionID", how = "left")
    df= df.sort_values("TransactionDT").reset_index(drop=True)

    out_path = out_dir / "train.parquet"
    df.to_parquet(out_path) #Faster than CSV

    print("Saved processed dataset to the following location: ", out_path)
    print("Shape: ", df.shape)

if __name__ == "__main__":
    main()

