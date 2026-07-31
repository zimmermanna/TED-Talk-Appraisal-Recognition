import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.MILArchitecture.MILDataset import MILDataset
from src.MILArchitecture.windows import mil_collate_fn


def initialize_loaders_per_split(test_source, sources, metadata, bags, labels, clip_ids,config,seed=42):
    # Transform Sources List into a NumPy Array
    sources = np.array(sources)

    # Splitting Remaining Data with 80/20 into Train/Val
    remaining_sources = sources[sources != test_source]
    train_speakers, val_speakers = train_test_split(remaining_sources, test_size=0.2, random_state=seed)

    # Getting Indices for Train, Val and Test
    train_indices = metadata[metadata["source_id"].isin(train_speakers)].index.tolist()
    val_indices = metadata[metadata["source_id"].isin(val_speakers)].index.tolist()
    test_indices = metadata[metadata["source_id"] == test_source].index.tolist()

    # Getting Bags and Labels based on Indices
    train_bags = [bags[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_ids = [clip_ids[i] for i in train_indices]

    val_bags = [bags[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_ids = [clip_ids[i] for i in val_indices]

    test_bags = [bags[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_ids = [clip_ids[i] for i in test_indices]

    # Normalizing Bags based on Train Data with RobustScaler
    all_train_windows = np.vstack(train_bags)

    scaler = RobustScaler()
    scaler.fit(all_train_windows)

    train_bags_scaled = [scaler.transform(bag) for bag in train_bags]
    val_bags_scaled = [scaler.transform(bag) for bag in val_bags]
    test_bags_scaled = [scaler.transform(bag) for bag in test_bags]

    # Bags to Tensor Dataset
    train_dataset = MILDataset(train_bags_scaled, train_labels, train_ids)
    val_dataset = MILDataset(val_bags_scaled, val_labels, val_ids)
    test_dataset = MILDataset(test_bags_scaled, test_labels, test_ids)

    # DataLoader with Padding + Masking
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=mil_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=mil_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=mil_collate_fn
    )

    return train_loader, val_loader, test_loader
