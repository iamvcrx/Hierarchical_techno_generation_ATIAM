from pydantic import BaseModel
import yaml
import os


# Define one class for each param family (defined from BaseModel)
class Model(BaseModel):
    n_latent: int = 128  # variable : type = valeur par défaut (si le fichier de config ne contient pas l'entrée)
    ratios: list
    model_name: str = None


class Train(BaseModel):
    lr: float = 128
    batch_size: int = 32


# Macro class containig each parameters families
class MainConfig(BaseModel):
    model: Model = None
    train: Train = None


#Loading the config file into the classes :

with open(os.path.join("config.yaml"), "r") as config_f:
    config_dict = yaml.safe_load(config_f)

    model_dict = {
        "model": Model(**config_dict["model"]),
        "train": Train(**config_dict["train"]),
    }

    main_config = MainConfig(**model_dict)

# You can then access your config params through the main_config object
print("n_latent : ", main_config.model.n_latent)
print("lr : ", main_config.train.lr)
print("batch_size : ", main_config.train.batch_size)

# After loading the config you can change the parameter values

main_config.model.model_name = "my_amazing_model"

# You can the save the model config for future model import
with open(os.path.join("train_config.yaml"), 'w') as s:
    yaml.safe_dump(main_config.dict(), stream=s)
