from sklearn.preprocessing import RobustScaler

def train_val_test_split(df):
    df_train = df[df['split'] == "train"]
    df_val = df[df['split'] == "validation"]
    df_test = df[df['split'] == "test"]

    return df_train, df_val, df_test


def robust_normalize(df_train, df_val, df_test):

    au_cols = [col for col in df_train.columns if col.startswith("AU") and col.endswith("_r")]
    scaler = RobustScaler()

    scaler.fit(df_train[au_cols])

    df_train[au_cols] = scaler.transform(df_train[au_cols])
    df_val[au_cols] = scaler.transform(df_val[au_cols])
    # df_test[au_cols] = scaler.transform(df_test[au_cols])

    return df_train, df_val, df_test


def apply_speaker_normalization(df, stats, au_cols):
    df = df.copy()

    for source_id in df["source_id"].unique():
        if source_id not in stats.index:
            continue

        mask = df["source_id"] == source_id

        for col in au_cols:
            median = stats.loc[source_id, f"{col}_median"]
            iqr = stats.loc[source_id, f"{col}_iqr"]

            if iqr == 0:
                iqr = 1

            df.loc[mask, col] = (df.loc[mask, col] - median) / iqr

    return df

def speaker_normalize(df_train, df_val, df_test):
    au_cols = [col for col in df_train.columns if col.startswith("AU") and col.endswith("_r")]

    stats = (
        df_train
        .groupby("source_id")[au_cols]
        .agg(["median", lambda x: x.quantile(0.75) - x.quantile(0.25)])
    )

    stats.columns = [
        f"{col}_{stat}" if stat != "<lambda_0>" else f"{col}_iqr"
        for col, stat in stats.columns
    ]

    df_train = apply_speaker_normalization(df_train, stats, au_cols)
    df_val = apply_speaker_normalization(df_val, stats, au_cols)
    # df_test = apply_speaker_normalization(df_test, stats, au_cols)

    return df_train, df_val, df_test