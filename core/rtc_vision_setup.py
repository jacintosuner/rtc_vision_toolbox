from camera.orbbec.ob_camera import OBCamera    
from robot.ros_robot import ROSRobot
from calibration import calibrations
from core import utils

from typing import Union
import numpy as np

CameraType = Union[OBCamera]
RobotType = Union[ROSRobot]

class RTCVisionSetup:
    """
    Initialize device drivers and cell configuration (like calibrations)
    """
    # data members
    cameras: dict[str, CameraType] = None
    robot: RobotType = None
    T_cam2robot: dict[str, np.ndarray] = None
    
    def __init__(self, *args):
        #TODO: implement this method
        
        if len(args) == 0:
            raise ValueError("No arguments provided for RTCVisionSetup")
        
        elif len(args) == 1 and type(args[0]) == str:
            # TODO: read configuration file
            pass
        
        #TODO:implement other initilization methods
        
        else:
            raise ValueError("Invalid arguments provided for RTCVisionSetup")
        
        pass
    
    def perform_robot_camera_calibration(self):
        """
        Perform robot-camera calibration
        """
        
        # calibrate each camera by collecting samples of marker poses
        for cam_id, camera in self.cameras.items():
            keep_going = 'y'
            print(f"Starting calibration for {cam_id} camera")
            if cam_id not in self.T_cam2robot:
                keep_going = input(f"Calibrate {cam_id} camera? (y/n): ")
            else:
                print(f"Calibration for {cam_id} camera already exists")
                keep_going = input(f"Recalibrate {cam_id} camera? (y/n): ")

            if keep_going.lower() == 'y':
                T_cam2robot = calibrations.get_robot_camera_tf(camera, self.robot, 'PLAY')
                self.T_cam2robot[cam_id] = T_cam2robot
                print(f"Calibration for {cam_id} camera completed")
                
        # check errors for all cameras
        avg, std = self.get_calibration_errors(self.T_cam2robot)
        
        # improve calibration
        keep_going = input("Improve calibration using PCD registration? (y/n): ")
        if keep_going.lower() == 'y':
            #TODO: implement PCD registration
            ...
            
            avg, std = self.get_calibration_errors(T_cam2robot)
            
            #TODO: visualize the merged point cloud
            ...
            
            accept = input("Accept calibration? (y/n): ")
            #TODO: implement accept calibration
            
        
    def get_calibration_errors(self, T_cam2robot) -> tuple[list[float], list[float]]:
        """
        TODO: Get calibration errors using latest sample data and T_cam2robot
        """
        ...
        
        print("Average errors: ", avg)
        print("Standard deviation of errors: ", std)
        pass