from pydantic import BaseModel
import yaml
import os


class VAE_raw(BaseModel):
    in_channels: int = 1
    n_latent: int
    ratios: list
    channel_size: list
    model_name: str = "VAE_raw"


    
class Dataset(BaseModel):
    batch_size: int = 32
    valid_ratio: float = 0.2
    num_thread: int = 0
    

class Train(BaseModel):
    lr: float
    beta: float = 0.1
    n_fft_l: list = [2048,1024,512,256]
    w: str = "Hamming"
    epochs: int 
    save_ckpt: int = 5
    add_fig_sound: int = 5
    loss: str = "MSE"

    

class MainConfig(BaseModel):
    vae_raw: VAE_raw = None
    train: Train = None
    dataset: Dataset = None


def load_config(yaml_filepath="config.yaml"):
    with open(yaml_filepath, "r") as config_f:
        try:
            config_dict = yaml.safe_load(config_f)
            model_dict = {
                "vae_raw": VAE_raw(**config_dict["vae_raw"]),
                "train": Train(**config_dict["train"]),
                "dataset": Dataset(**config_dict["dataset"])

            }
            main_config = MainConfig(**model_dict)
            return main_config
        
        except yaml.YAMLError as e:
            print(e)


def save_config(main_config, config_name="train_test_config.yaml"):
    with open(os.path.join(config_name), 'w') as s:
        yaml.safe_dump(main_config.dict(), stream=s)