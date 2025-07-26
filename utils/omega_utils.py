from datetime import datetime
import shutil
import sys
from omegaconf import OmegaConf
import os

def load_config(save=False, config_path="config.yaml") -> OmegaConf:
    print(sys.argv)
    print('loading conf:',config_path)
    config = OmegaConf.load(config_path)
    metadata = config.metadata
    metadata.name = metadata.name.replace(" ", "_").lower()  # Normalize the experiment name
    expr_name = metadata.name
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")  # add hours and minutes
    expr_folder = f"experiments/{expr_name}/{date}"
    if 'experiment_folder' not in metadata:
        metadata.date = date
        metadata.experiment_folder = expr_folder
        if save:
            os.makedirs(expr_folder, exist_ok=True)
            OmegaConf.save(config, f'{expr_folder}/config.yaml')
    
    return config


if __name__ == "__main__":
    config = load_config()
    print(OmegaConf.to_yaml(config))
    # Copy the config file to the experiment folder
    print(f"Config saved to {config.metadata.experiment_folder}/config.yaml")