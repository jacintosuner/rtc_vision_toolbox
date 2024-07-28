from rtc_core.place_skill.place_teach import TeachPlace
from rtc_core.place_skill.place_learn import LearnPlace
from rtc_core.place_skill.place_execute import ExecutePlace

from omegaconf import DictConfig, OmegaConf

class place_skill(TeachPlace, LearnPlace, ExecutePlace):
    def __init__(self, config: DictConfig):
        TeachPlace.__init__(self, config)
        LearnPlace.__init__(self, config)
        ExecutePlace.__init__(self, config)