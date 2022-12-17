from pydantic import BaseModel
import yaml
import os


class Dataset_Hier(BaseModel):
    batch_size: int = 32
    valid_ratio: float = 0.2
    num_thread: int = 0


class Train_hier(BaseModel):
    lr: float
    beta: float = 0.1
    epochs: int 
    save_ckpt: int = 5
    add_fig_sound: int = 5
    loss_type: str = "mean"

class VAE_Hier(BaseModel):
    n_latent: int
    ratios: list
    channel_size: list
    model_name: str = "VAE_hier"

    

class MainConfig(BaseModel):
    dataset_hier: Dataset_Hier = None
    vae_hier: VAE_Hier = None
    train_hier: Train_hier = None


            
def load_config_hier(yaml_filepath="config.yaml"):
    with open(yaml_filepath, "r") as config_f:
        try:
            config_dict = yaml.safe_load(config_f)
            model_dict = {
                "dataset_hier": Dataset_Hier(**config_dict["dataset_hier"]),
                "vae_hier": VAE_Hier(**config_dict["vae_hier"]),
                "train_hier": Train_hier(**config_dict["train_hier"])

            }
            main_config = MainConfig(**model_dict)
            return main_config
        
        except yaml.YAMLError as e:
            print(e)

def save_config(main_config, config_name="train_test_config.yaml"):
    with open(os.path.join(config_name), 'w') as s:
        yaml.safe_dump(main_config.dict(), stream=s)