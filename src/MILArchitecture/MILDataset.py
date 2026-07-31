import torch
from torch.utils.data import Dataset

class MILDataset(Dataset):
    def __init__(self, bags, labels, clip_ids):
        self.bags = [torch.tensor(bag, dtype=torch.float32) for bag in bags]
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.clip_ids = clip_ids

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        label = self.labels[index]
        clip_id = self.clip_ids[index]

        return bag, label, clip_id