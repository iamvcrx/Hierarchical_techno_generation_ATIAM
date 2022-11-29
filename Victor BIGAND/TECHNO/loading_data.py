import torch
from torch.utils.data import Dataset, Subset
import os
#import torchaudio
import numpy as np


class TechnoDataset(Dataset):

    def __init__(self,
                 dat_location,
                 size=2**15) -> None:
        super().__init__()

        self.samples = np.memmap(
            dat_location,
            dtype="float32",
            mode="r",
        )
        self.samples = self.samples[:size * (len(self.samples) // size)]
        self.samples = self.samples.reshape(-1, 1, size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.from_numpy(np.copy(self.samples[index])).float()

dataset = TechnoDataset("techno_resampled.dat")

print(dataset[0])