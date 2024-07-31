import argparse
import os
import sys

import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def get_taxpose_config(config_path, config_name):
    
    conf: DictConfig = None
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    relative_config_path = os.path.relpath(config_path, script_dir)
    print(relative_config_path)
    
    breakpoint()
    GlobalHydra.instance().clear()
    @hydra.main(config_path=relative_config_path, version_base=None)
    def config(config: DictConfig):
        nonlocal conf
        raw_conf = compose(config_name=config_name)        
        resolved_conf = OmegaConf.to_container(raw_conf, resolve=True)
        resolved_conf = OmegaConf.create(resolved_conf)
        conf = resolved_conf
    
    config()
    print(OmegaConf.to_yaml(conf, resolve=True))    
    
    GlobalHydra.instance().clear()
    with initialize(config_path="../data/demonstrations/07-24-wp"):
        conf2 = compose(config_name="place_object.yaml")
    
    config = OmegaConf.merge(conf, conf2)
    
    breakpoint()
    

@hydra.main(config_path="../models/taxpose/configs/", version_base=None)
def get_taxpose_config_2(config: DictConfig):
    config: DictConfig = compose(config_name="commands/mfi/waterproof/eval_taxpose_06-20-wp_place.yaml")
        
    print(OmegaConf.to_yaml(config, resolve=True))
    
    
if __name__ == "__main__":
    # Custom argparse handling
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args, unknown = parser.parse_known_args()

    # Filter out custom arguments before Hydra processes them
    filtered_argv = [arg for arg in sys.argv if arg.startswith("--hydra")]

    # Manually call Hydra main
    sys.argv = [sys.argv[0]] + filtered_argv    


    get_taxpose_config("/home/mfi/repos/rtc_vision_toolbox/models/taxpose/configs/", "commands/mfi/waterproof/eval_taxpose_06-20-wp_place.yaml")    
    # get_taxpose_config_2()
    