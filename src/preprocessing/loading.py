import glob
import pandas as pd
import os

"""
Loading the raw openface CSVs and converting them into a dataframe. Then combining it with the related labels.
Returns: Combined Dataframe
"""
def load_df():
    print('=' * 5, "Loading Data")

    # Loading Paths
    files = glob.glob("../data/openface/*.csv")
    labels = pd.read_csv("../data/labels.csv")

    data = []

    for file in files:
        clip_id = os.path.splitext(os.path.basename(file))[0]
        df_local = pd.read_csv(file)

        # Deleting Whitespaces in Column Names
        df_local.columns = df_local.columns.str.strip()

        # Assigning the ClipID to OpenFace Data
        df_local = df_local.copy()
        df_local = df_local.assign(clip_id=clip_id)

        data.append(df_local)

    df = pd.concat(data, ignore_index=True)

    # Combining OpenFace Data with Labels
    df = df.merge(labels, on="clip_id", how="left")

    return df