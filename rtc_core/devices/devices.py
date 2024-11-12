import argparse
import logging
import os
import time

import cv2
import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf

try:
    from camera.orbbec.ob_camera import OBCamera
except ImportError:
    logging.warning("OBCamera is not available.")

try:
    from camera.rs_ros.rs_ros import RsRos
except ImportError:
    logging.warning("RsRos is not available.")

try:
    from camera.zed_ros.zed_ros import ZedRos
except ImportError:
    logging.warning("ZedRos is not available.")

try:
    from robot.robotiq.robotiq_gripper import RobotiqGripper
except ImportError:
    logging.warning("RobotiqGripper is not available.")
    
try:
    from robot.ros_robot.ros_robot import ROSRobot
except ImportError:
    logging.warning("ROSRobot is not available.")


class Devices:

    def __init__(self, conf: DictConfig, debug: bool = False):
        self.__cam = {}
        self.__robot = None
        self.__gripper = None

        # Initialize cameras
        for key in conf.cameras.keys():
            match conf.cameras[key]['class']:
                case "ZedRos":
                    self.__cam[key] = ZedRos(
                        camera_node=conf.cameras[key].init_args.camera_node,
                        camera_type=conf.cameras[key].init_args.camera_type,
                        rosmaster_ip=conf.cameras[key].init_args.rosmaster_ip if conf.cameras[
                            key].init_args.rosmaster_ip is not None else "localhost",
                        rosmaster_port=conf.cameras[key].init_args.rosmaster_port if conf.cameras[
                            key].init_args.rosmaster_port is not None else 9090,
                        debug=debug
                    )
                case "RsRos":
                    self.__cam[key] = RsRos(
                        camera_node=conf.cameras[key].init_args.camera_node,
                        camera_type=conf.cameras[key].init_args.camera_type,
                        rosmaster_ip=conf.cameras[key].init_args.rosmaster_ip if conf.cameras[
                            key].init_args.rosmaster_ip is not None else "localhost",
                        rosmaster_port=conf.cameras[key].init_args.rosmaster_port if conf.cameras[
                            key].init_args.rosmaster_port is not None else 9090,
                        debug=debug
                    )
                case "OBCamera":
                    self.__cam[key] = OBCamera(
                        serial_no=conf.cameras[key].init_args.serial_number
                    )
                case _:
                    raise NotImplementedError(
                        f"Camera type {conf.cameras[key].type} is not supported."
                    )

        # Initialize robot
        self.__robot = ROSRobot(
            robot_name=conf.robot.init_args.robot_name,
            rosmaster_ip=conf.robot.init_args.rosmaster_ip,
        )

        # Initialize gripper
        self.__gripper = RobotiqGripper(portname=conf.gripper.init_args.port)

    def robot_get_eef_pose(self) -> np.ndarray:
        return self.__robot.get_eef_pose()

    def robot_move_to_pose(self, pose: np.ndarray, 
                           max_velocity_scaling_factor=0.3, 
                           max_acceleration_scaling_factor=0.3) -> None:
        for _ in range(3):
            if not self.__robot.move_to_pose(
                position=pose[:3, 3], orientation=pose[:3, :3],
                max_velocity_scaling_factor=max_velocity_scaling_factor,
                max_acceleration_scaling_factor=max_acceleration_scaling_factor
            ):
                print(f"Failed to move to pose.")
            else:
                return True
        return False

    def gripper_open(self) -> None:
        self.__gripper.openGripper()

    def gripper_close(self) -> None:
        self.__gripper.closeGripper()

    def cam_get_rgb_image(self, cam_name: str) -> np.ndarray:
        return self.__cam[cam_name].get_rgb_image()

    def cam_get_raw_depth_data(self, cam_name: str, max_depth: int) -> np.ndarray:
        if max_depth is None:
            return self.__cam[cam_name].get_raw_depth_data()
        else:
            return self.__cam[cam_name].get_raw_depth_data(max_depth=max_depth)

    def cam_get_colormap_depth_image(self, cam_name: str, max_depth: int) -> np.ndarray:
        if max_depth is None:
            return self.__cam[cam_name].get_colormap_depth_image()
        else:
            return self.__cam[cam_name].get_colormap_depth_image(max_depth=max_depth)

    def cam_get_point_cloud(self, cam_name: str, max_mm: int) -> o3d.geometry.PointCloud:
        if max_mm is None:
            return self.__cam[cam_name].get_point_cloud()
        else:
            return self.__cam[cam_name].get_point_cloud(max_mm=max_mm)
