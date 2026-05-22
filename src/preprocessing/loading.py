import glob
import pandas as pd
import os

def load_df():
    """
    Loading the raw openface CSVs and converting them into a dataframe. Then combining it with the related labels.
    Returns: Combined Dataframe

    """
    print('-' * 10, "Loading Data", '-' * 10)
    files = glob.glob("../data/raw/openface/*.csv")
    labels = pd.read_csv("../data/labels.csv")

    data = []

    for file in files:
        df_local = pd.read_csv(file)

        clip_id = os.path.splitext(os.path.basename(file))[0]
        df_local["clip_id"] = clip_id

        data.append(df_local)

    df = pd.concat(data, ignore_index=True)
    df = df.merge(labels, on="clip_id", how="left")

    # Deleting whitespaces in column names
    df.columns = df.columns.str.strip()

    return df