from src.utils.data import au_cols, sources


def normalize_by_speaker(df, iqr_limit=0.1, debug=False):
    df = df.copy()
    small_iqr = []

    # Normalizing per Speaker/Source
    for source_id in sources:
        mask = df["source_id"] == source_id

        # Normalizing per AU
        for col in au_cols:
            median = df.loc[mask, col].median()
            q75 = df.loc[mask, col].quantile(0.85)
            q25 = df.loc[mask, col].quantile(0.25)
            iqr = q75 - q25

            # Avoiding Overinterpretation by Capping too small IQR
            if iqr < iqr_limit:
                small_iqr.append(col)  # Documenting Which AU's are Subsceptible for Small IQR's
                iqr = iqr_limit

            # Updating Original Value by Normalized Value
            df.loc[mask, col] = (
                                        df.loc[mask, col] - median
                                ) / iqr

    # Documenting how many IQR Values were Capped
    if debug:
        print(f"{len(small_iqr)} IQR value(s) were transformed to {iqr_limit}: ")
        print(small_iqr)

    return df

def quality_filter(df):
    print('=' * 5, "Start Filtering for Confidence")
    print(f"Before filtering: {df.shape[0]} frames")

    df = df[df['confidence'] > 0.8]
    df = df[df['success'] == 1]

    print(f"After filtering: {df.shape[0]} frames")
    print('=' * 5, "Finished Filtering")

    return df