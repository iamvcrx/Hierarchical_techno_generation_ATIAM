import torch
from torch.utils.tensorboard import SummaryWriter
from configs import config
import dataset.Techno_Dataset as dataset
import models.Encoder
import models.Decoder
import models.VAE_raw



# Définit le device sur lequel on va train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Permet de load toute la configuration
main_config = config.load_config("Quentin_Le_Gall/main/config.yaml")

# Import du dataset
train_loader,valid_loader = dataset.Create_Dataset(dataset_dir="Quentin_Le_Gall/main/dataset/techno_resampled.dat",valid_ratio =main_config.dataset.valid_ratio,num_threads =main_config.dataset.num_thread,batch_size=main_config.dataset.batch_size)

# Création session tensorboard et save de la config
checkpoint = f"{main_config.vae_raw.model_name}"
writer = SummaryWriter('runs/' + checkpoint)


writer.add_text('VAE_raw_parameters', str(main_config.vae_raw))
writer.add_text('Train_parameters', str(main_config.train))
writer.add_text('Dataset_config', str(main_config.dataset))


# Save le config file pour pouvoir le rouvrir par la suite (save dans le dossier de logs runs/model_name)
#config_path = f"Quentin_Le_Gall/main/runs/{main_config.vae_raw.model_name}"
#config_name = f"{config_path}/{main_config.vae_raw.model_name}_train_config.yaml"
#config.save_config(main_config , config_name)


encoder = models.Encoder.Encoder(main_config.vae_raw.in_channels,main_config.vae_raw.n_latent,main_config.vae_raw.ratios,main_config.vae_raw.channel_size)
decoder = models.Decoder.Decoder(main_config.vae_raw.in_channels,main_config.vae_raw.n_latent,main_config.vae_raw.ratios,main_config.vae_raw.channel_size)
model = models.VAE_raw.VAE_raw(encoder,decoder)






