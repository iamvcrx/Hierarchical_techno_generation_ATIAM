import torch
from torch.utils.tensorboard import SummaryWriter
from configs import config_hierarchical
from configs import config
import dataset.Techno_Dataset as dataset
import models.Encoder_hierarchical
import models.Decoder_hierarchical
import models.VAE_hierarchical
import train.Train_hierarchical
import models.Encoder
import models.Decoder
import models.VAE_raw
import train.Train
from torchsummary import summary
from torch.utils.data import TensorDataset,Dataset
import numpy as np
import os




run_name = "Prepa_Run"
path_trained_model = "/slow-1/atiam/hierarchical_vae/Quentin/Hierarchical_techno_generation_ATIAM/main/runs"
path_dataset = "/fast-1/atiam/hierarchical_vae/techno_resampled.dat"



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device ='cpu'

# Permet de load toute la configuration
main_config_raw = config.load_config("{}/{}/{}_train_config.yaml".format(path_trained_model,run_name,run_name))
# Import du dataset

train_loader,valid_loader = dataset.Create_Dataset(dataset_dir= path_dataset,
                                                   valid_ratio = main_config_raw.dataset.valid_ratio,
                                                   num_threads = main_config_raw.dataset.num_thread,
                                                   batch_size = main_config_raw.dataset.batch_size)

encoder = models.Encoder.Encoder(main_config_raw.vae_raw.in_channels,
                                 main_config_raw.vae_raw.n_latent,
                                 main_config_raw.vae_raw.ratios,
                                 main_config_raw.vae_raw.channel_size).to(device)

decoder = models.Decoder.Decoder(main_config_raw.vae_raw.in_channels,
                                 main_config_raw.vae_raw.n_latent,
                                 main_config_raw.vae_raw.ratios,
                                 main_config_raw.vae_raw.channel_size).to(device)
                
model_raw = models.VAE_raw.VAE_raw(encoder,decoder).to(device)




### Load modèle entrainé ###
Dict = torch.load("{}/{}/{}.pt".format(path_trained_model,run_name,run_name)) 
model_dict = Dict["VAE_model"]
model_raw.load_state_dict(model_dict)
model_raw.eval()



main_config_hier = config_hierarchical.load_config_hier("./config_hierarchical.yaml")


train_loader_latent, valid_loader_latent = dataset.Create_Latent_Dataset(dataset_dir = path_dataset,
                                                                         valid_ratio = main_config_raw.dataset.valid_ratio,
                                                                         num_threads = main_config_raw.dataset.num_thread,
                                                                         raw_batch_size = main_config_raw.dataset.batch_size,
                                                                         latent_batch_size = main_config_hier.dataset_hier.batch_size,
                                                                         model_raw = model_raw,
                                                                         device = device)



main_config_hier = config_hierarchical.load_config_hier("./config_hierarchical.yaml")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_raw.to(device)

encoder = models.Encoder_hierarchical.Encoder_hierarchical(main_config_raw.vae_raw.n_latent*2,
                                                           main_config_hier.vae_hier.n_latent,
                                                           main_config_hier.vae_hier.ratios,
                                                           main_config_hier.vae_hier.channel_size).to(device)

decoder = models.Decoder_hierarchical.Decoder_hierarchical(main_config_raw.vae_raw.n_latent*2,
                                                           main_config_hier.vae_hier.n_latent,
                                                           main_config_hier.vae_hier.ratios,
                                                           main_config_hier.vae_hier.channel_size).to(device)
                
model_hier = models.VAE_raw.VAE_raw(encoder,decoder).to(device)

print("\n")
print(summary(model_hier,(256,128)))
print("\n")

checkpoint = "{}".format(main_config_hier.vae_hier.model_name)
writer = SummaryWriter("{}/{}/".format(path_trained_model,run_name) + checkpoint)


writer.add_text('VAE_hier_parameters', str(main_config_hier.vae_hier))
writer.add_text('Train_hier_parameters', str(main_config_hier.train_hier))
writer.add_text('Dataset_hier_config', str(main_config_hier.dataset_hier))

# Save le config file pour pouvoir le rouvrir par la suite (save dans le dossier de logs runs/model_name)
config_path = "{}/{}/{}".format(path_trained_model,run_name, main_config_hier.vae_hier.model_name)   
config_name = "{}/{}_train_config.yaml".format(config_path, main_config_hier.vae_hier.model_name)   
config_hierarchical.save_config(main_config_hier , config_name)


VAE_train = train.Train_hierarchical.train_VAE_hierarchical(train_loader_latent = train_loader_latent,
                                                            valid_loader_latent = valid_loader_latent,
                                                            model_raw = model_raw,
                                                            model_hierarchical = model_hier,
                                                            run_name = run_name,
                                                            latent_dim = main_config_hier.vae_hier.n_latent,
                                                            lr = main_config_hier.train_hier.lr,
                                                            beta = main_config_hier.train_hier.beta,
                                                            model_name = main_config_hier.vae_hier.model_name,
                                                            num_epochs = main_config_hier.train_hier.epochs,
                                                            writer = writer,
                                                            save_ckpt = main_config_hier.train_hier.save_ckpt,
                                                            path_trained_model = path_trained_model,
                                                            add_figure_sound = main_config_hier.train_hier.add_fig_sound,
                                                            device = device,
                                                            valid_loader_raw = valid_loader)



VAE_train.train_step()