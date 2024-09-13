import argparse
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from rtc_core.place_skill.place_teach import TeachPlace

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    
    # Read configuration file
    config_file = args.config
    config_dir = os.path.dirname(config_file)
    config_name = os.path.basename(config_file)
    
    print(f"Reading configuration file: {config_file}")
    print(f"Configuration directory: {config_dir}")
    print(f"Configuration name: {config_name}")
    
    hydra.initialize(config_path=config_dir, version_base="1.3")
    config: DictConfig = hydra.compose(config_name)
      
    place_teach = TeachPlace(config)
    place_teach.collect_demonstrations()
