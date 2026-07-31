def simple_aggregation(df):
    au_cols = [col for col in df.columns if col.startswith("AU") and col.endswith("_r")]

    df_grouped = df.groupby('clip_id').agg({
        **{au: "mean" for au in au_cols},

        "label" : "first",
        "source_id": "first"
    })

    return df_grouped

def advanced_aggregation(df):
    au_cols = [col for col in df.columns if col.startswith("AU") and col.endswith("_r")]

    def q95(x):
        return x.quantile(0.95)

    df_grouped = df.groupby("clip_id").agg({
        **{au: ["mean", "std", q95] for au in au_cols},

        "label": "first",
        "source_id": "first"
    })

    df_grouped.columns = [
        "_".join(col).strip("_") for col in df_grouped.columns
    ]

    df_grouped = df_grouped.rename(columns={
        "label_first": "label",
        "source_id_first": "source_id"
    })

    return df_grouped