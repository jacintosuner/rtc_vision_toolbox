from core.rtc_vision_setup import RTCVisionSetup
from core.rtc_vision_model import RTCVisionModel
from core import utils

class RTCVisionExecute():
    """
    Perform task using rtc_vision trained skill(s)
    """
    
    # data members
    setup: RTCVisionSetup = None
    models: dict[str, RTCVisionModel] = None
    
    def __init__(self, config_file=None):
        pass
    