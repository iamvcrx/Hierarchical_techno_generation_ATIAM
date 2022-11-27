import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, Subset
import os
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from helper_plot import hdr_plot_style
hdr_plot_style()


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



def Create_Dataset(dataset_dir = "./data/TECHNO/techno_resampled.dat", valid_ratio = 0.2,num_threads = 0,batch_size  = 32):
    dataset = TechnoDataset(dataset_dir)
    # Load the dataset for the training/validation sets
    train_valid_dataset =  dataset
    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset) +1)
    nb_valid =  int(valid_ratio * len(train_valid_dataset))

    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])


    # Prepare 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)


    print("The train set contains {} samples of length 32768, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} samples of length 32768, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))

    return train_loader,valid_loader