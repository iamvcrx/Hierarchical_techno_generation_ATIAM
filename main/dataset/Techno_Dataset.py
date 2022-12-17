import torch
from torch.utils.data import TensorDataset,Dataset
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



def Create_Dataset(dataset_dir = "./data/TECHNO/techno_resampled.dat", valid_ratio = 0.2,num_threads = 0,batch_size  = 32):
    dataset = TechnoDataset(dataset_dir)

    # Load the dataset for the training/validation sets
    train_valid_dataset =  dataset
    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset) +1)
    nb_valid =  int(valid_ratio * len(train_valid_dataset))

    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid],generator = torch.Generator().manual_seed(42))


    # Prepare 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)

  
    print("The train set contains {} samples of length 32768, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} samples of length 32768, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))

    return train_loader,valid_loader


def Create_Latent_Dataset(dataset_dir = "/fast-1/atiam/hierarchical_vae/techno_resampled.dat", 
                          valid_ratio = 0.05, 
                          num_threads = 0,
                          raw_batch_size  = 64,
                          latent_batch_size = 64,
                          model_raw = None,
                          device = 'cpu'):
    
    train_loader,valid_loader = Create_Dataset(dataset_dir = dataset_dir,
                                                       valid_ratio = valid_ratio,
                                                       num_threads = num_threads,
                                                       batch_size = raw_batch_size)
    """ with torch.no_grad():
        for i,batch in enumerate(train_loader):
            batch = batch.to(device)
             
    
            if i==0:
                mu,sigma = model_raw.encoder(batch)
                #print("mu :{}, shape: {}".format(mu.type,mu.shape))
                mu_sigma = torch.cat((mu,sigma),dim = 1)
                #print("mu_sigma :{}, shape: {}".format(mu_sigma.type,mu_sigma.shape))
                tensor_dataset = mu_sigma
            else :
                mu,sigma = model_raw.encoder(batch)
                mu_sigma = torch.cat((mu,sigma),dim = 1)
                tensor_dataset = torch.cat((tensor_dataset,mu_sigma),dim=0)
                #print("tensor_dataset :{}, shape: {}".format(tensor_dataset.type,tensor_dataset.shape))
            print(i) """
    tensor_dataset = []
    with torch.no_grad():
        for i,batch in enumerate(train_loader):
            batch = batch.to(device)
            mu,sigma = model_raw.encoder(batch)
            mu_sigma = torch.cat((mu,sigma),dim = 1)
            tensor_dataset.append(mu_sigma)

        for i,batch in enumerate(valid_loader):
            batch = batch.to(device)
            mu,sigma = model_raw.encoder(batch)
            mu_sigma = torch.cat((mu,sigma),dim = 1)
            tensor_dataset.append(mu_sigma)
            


    
    
    tensor_dataset = torch.cat(tensor_dataset,dim = 0)
    
    dataset = TensorDataset(tensor_dataset)
    # Load the dataset for the training/validation sets
    train_valid_dataset =  dataset
    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset) + 1) # + 1
    nb_valid =  int(valid_ratio * len(train_valid_dataset))

    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid],generator = torch.Generator().manual_seed(42))


    # Prepare 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=latent_batch_size, num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=latent_batch_size, shuffle=False, num_workers=num_threads)


    print("The latent train set contains {} samples of length 32768, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The latent validation set contains {} samples of length 32768, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))

    return train_loader,valid_loader
