def aggregate(df):
    au_cols = [col for col in df.columns if col.startswith("AU") and col.endswith("_r")]

    df_grouped = df.groupby('clip_id').agg({
        **{au: "mean" for au in au_cols},

        "confidence": "mean",
        "success": "first",
        "label" : "first",
        "source_id": "first",
        "duration": "first",
        "split": "first"
    })

    return df_grouped