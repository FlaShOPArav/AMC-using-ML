import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class RadioMLDataset(Dataset):
    def __init__(self, path="data/raw/RML2016.10a_dict.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        X, y = [], []
        mods = sorted(set(k[0] for k in data.keys()))

        for label, mod in enumerate(mods):
            for snr in range(-20, 20, 2):
                samples = data[(mod, snr)]
                X.append(samples)
                y.extend([label] * samples.shape[0])

        X = np.vstack(X)
        X = X / np.max(np.abs(X))

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.num_classes = len(mods)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
