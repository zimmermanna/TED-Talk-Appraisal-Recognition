import pandas as pd
import numpy as np

from src.configuration.config import LOCKED_CONFIG
from src.MILArchitecture.windows import initialize_all_bags

def quality_filter(df):
    print('=' * 5, "Start Filtering for Confidence")
    print(f"Before filtering: {df.shape[0]} frames")

    df = df[df['confidence'] > 0.8]
    df = df[df['success'] == 1]

    print(f"After filtering: {df.shape[0]} frames")
    print('=' * 5, "Finished Filtering")

    return df

def simple_summarize(df):
    au_cols = [c for c in df.columns if c.startswith("AU")]

    summary = pd.DataFrame({
        "Min": df[au_cols].min(),
        "Mean": df[au_cols].mean(),
        "Max": df[au_cols].max(),
        "Std": df[au_cols].std(),
        "Median": df[au_cols].median()
    })

    return summary.round(3)

def advanced_summarize(df):
    au_cols = [c for c in df.columns if c.startswith("AU")]

    grouped = df.groupby("source_id")[au_cols]

    summary = pd.concat({
        "Min": grouped.min(),
        "Mean": grouped.mean(),
        "Max": grouped.max(),
        "Std": grouped.std(),
        "Median": grouped.median()
    }, axis=1)

    summary = (
        summary
        .stack(level=1, future_stack=True)
        .rename_axis(index=["source_id", "Action Unit"])
    )

    return summary.round(3)

def label_distribution(df):
    summary_dataset = (
        df.groupby(["source_id", "label"])["clip_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    summary_dataset.columns = ["Label 0", "Label 1"]
    summary_dataset["Total"] = summary_dataset.sum(axis=1)

    summary_dataset["Label 0 (%)"] = (
            summary_dataset["Label 0"] / summary_dataset["Total"] * 100
    ).round(1)

    summary_dataset["Label 1 (%)"] = (
            summary_dataset["Label 1"] / summary_dataset["Total"] * 100
    ).round(1)

    return summary_dataset

def count_appraisal(df):
    appraisal_df = (
        df.groupby(["source_id", "clip_id"])["appraisal"]
        .first()
        .value_counts()
    )

    return appraisal_df

def window_statistics(df):
    bags, _, _, _, _ = initialize_all_bags(df, "AU02_r", LOCKED_CONFIG)

    window_counts = [len(bag) for bag in bags]

    print(f"Number of bags: {len(bags)}")
    print(f"Average windows per bag: {np.mean(window_counts):.2f}")
    print(f"Minimum windows per bag: {np.min(window_counts)}")
    print(f"Maximum windows per bag: {np.max(window_counts)}")