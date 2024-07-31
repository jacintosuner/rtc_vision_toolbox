import hydra
import numpy as np
import omegaconf


@hydra.main(config_path="../models/taxpose/configs", config_name="eval_mfi")
class 