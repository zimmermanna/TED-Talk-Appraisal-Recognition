import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


def initialize_all_bags(df, au_cols, config, log=False):
    # Converts a single AU column into a list
    if isinstance(au_cols, str):
        au_cols = [au_cols]

    bags = []
    labels = []
    clip_ids = []
    source_ids = []

    # Iterating Through All Clip_IDs
    for clip_id, group in df.groupby("clip_id"):
        bag = create_bag(group, au_cols, config)
        label = group["label"].iloc[0]
        source_id = group["source_id"].iloc[0]

        bags.append(bag)
        labels.append(label)
        clip_ids.append(clip_id)
        source_ids.append(source_id)

    # Collecting Metadata as Detailed Summary
    metadata = pd.DataFrame({
        "clip_id": clip_ids,
        "source_id": source_ids,
        "label": labels,
        "num_windows": [bag.shape[0] for bag in bags]
    })

    # Print Initialization Summary
    if (log):
        print(f"{len(bags)} bags were successfully initialized, each with {bags[0].shape[1]} Features.")

        print("    * Min windows per Clip:", metadata["num_windows"].min())
        print("    * Max windows per Clip:", metadata["num_windows"].max())
        print("    * Average windows per Clip:", metadata["num_windows"].mean().round(2))
        print("See more details in 'metadata'.")

    return bags, labels, clip_ids, source_ids, metadata

def create_bag(group, au_cols, config):
    windows = create_windows(group, config)

    window_features = []

    for window in windows:
        features = extract_window_features(window, au_cols)
        window_features.append(features)

    bag = np.vstack(window_features)
    return bag

def create_windows(group, config):
    window_size = config["window_size"]
    stride = config["stride"]

    windows = []
    n = len(group)

    # Creating at least one Window if Clip is too short
    if n <= window_size:
        windows.append(group)
        return windows

    # Creating half-overlapping Windows
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(group.iloc[start:end])

    # Last Window size can be more flexible
    last_window = group.iloc[n - window_size:n]
    if not windows[-1].index.equals(last_window.index):
        windows.append(last_window)

    return windows

def extract_window_features(window, au_cols):
    features = []

    for col in au_cols:
        values = window[col]
        features.append(values.mean())

    return np.array(features)


# Function adds Padding-Windows inside a bag.
# Per batch, the maximum amount of windows is chosen. All bags with less window are filled with false windows to maintain an equal amount of windows.
def mil_collate_fn(batch):
    bags, labels, clip_ids = zip(*batch)

    # Amount of windows per clip
    lengths = torch.tensor([bag.shape[0] for bag in bags])

    # Add Padding to bag to maintain max amount of windows
    # Mask: 1 -> real window, 0 -> Padding
    padded_bags = pad_sequence(bags, batch_first=True)
    max_len = padded_bags.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    labels = torch.stack(labels)

    return padded_bags, labels, mask, clip_ids