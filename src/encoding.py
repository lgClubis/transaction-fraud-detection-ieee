import pandas as pd

def target_encode(train_df, val_df, col, target, min_samples=50):
    global_mean = train_df[target].mean()

    stats = (
        train_df
        .groupby(col)[target]
        .agg(["mean", "count"])
    )
    #Prevent overfitting
    stats["te"] = stats.apply(
        lambda r: r["mean"] if r["count"] >= min_samples else global_mean,
        axis=1
    )

    mapping = stats["te"]

    train_enc = train_df[col].map(mapping).fillna(global_mean)
    val_enc = val_df[col].map(mapping).fillna(global_mean)

    return train_enc, val_enc
