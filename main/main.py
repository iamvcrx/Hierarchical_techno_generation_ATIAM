import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
# Import des fichiers du modele
import configs.config
import dataset.Techno_Dataset as dataset
import models.Encoder
import models.Decoder
import models.VAE_raw
import train.Train


############################################################# A MODIF #########################
path_main = "/slow-1/atiam/hierarchical_vae/Quentin/Hierarchical_techno_generation_ATIAM/main"
path_dataset = "/fast-1/atiam/hierarchical_vae/techno_resampled.dat"
############################################################# A MODIF #########################


# Definit le device sur lequel on va train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Permet de load toute la configuration
main_config = configs.config.load_config("{}/config.yaml".format(path_main))

# Import du dataset
train_loader,valid_loader = dataset.Create_Dataset(dataset_dir=path_dataset,
                                                   valid_ratio =main_config.dataset.valid_ratio,
                                                   num_threads =main_config.dataset.num_thread,
                                                   batch_size=main_config.dataset.batch_size)

# Creation session tensorboard et save de la config
checkpoint = "{}".format(main_config.vae_raw.model_name)
writer = SummaryWriter("{}/runs/".format(path_main) + checkpoint)


writer.add_text('VAE_raw_parameters', str(main_config.vae_raw))
writer.add_text('Train_parameters', str(main_config.train))
writer.add_text('Dataset_config', str(main_config.dataset))


# Save le config file pour pouvoir le rouvrir par la suite (save dans le dossier de logs runs/model_name)
config_path = "{}/runs/{}".format(path_main, main_config.vae_raw.model_name)   #f"{path_main}/runs/{main_config.vae_raw.model_name}"        
config_name = "{}/{}_train_config.yaml".format(config_path, main_config.vae_raw.model_name)     #f"{config_path}/{main_config.vae_raw.model_name}_train_config.yaml"
configs.config.save_config(main_config , config_name)


encoder = models.Encoder.Encoder(main_config.vae_raw.in_channels,
                                 main_config.vae_raw.n_latent,
                                 main_config.vae_raw.ratios,
                                 main_config.vae_raw.channel_size).to(device)

decoder = models.Decoder.Decoder(main_config.vae_raw.in_channels,
                                 main_config.vae_raw.n_latent,
                                 main_config.vae_raw.ratios,
                                 main_config.vae_raw.channel_size).to(device)
                       
model = models.VAE_raw.VAE_raw(encoder,decoder).to(device)

print(summary(model,(1,32768)))


VAE_train = train.Train.train_VAE(train_loader,
                                  model,
                                  writer,
                                  main_config.vae_raw.n_latent,
                                  main_config.train.w,
                                  main_config.train.lr,
                                  main_config.train.n_fft_l,
                                  main_config.train.beta,
                                  main_config.vae_raw.model_name,
                                  main_config.train.epochs,
                                  main_config.train.save_ckpt,
                                  path_main=path_main,
                                  add_figure_sound=main_config.train.add_fig_sound,
                                  loss = main_config.train.loss,
                                  device = device)

VAE_train.train_step()

