import argparse
import os
import time

import cv2
import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf

from camera.orbbec.ob_camera import OBCamera
from camera.rs_ros.rs_ros import RsRos
from camera.zed_ros.zed_ros import ZedRos
from robot.robotiq.robotiq_gripper import RobotiqGripper
from robot.ros_robot.ros_robot import ROSRobot


class Devices:

    def __init__(self, conf: DictConfig):
        self.__cam = {}
        self.__robot = None
        self.__gripper = None

        # Initialize cameras
        for key in conf.cameras.keys():
            match conf.cameras[key].type:
                case "ZedRos":
                    self.__cam[key] = ZedRos(
                        camera_node=conf.cameras[key].init_args.camera_node,
                        camera_type=conf.cameras[key].init_args.camera_type,
                        rosmaster_ip=conf.cameras[key].init_args.rosmaster_ip if conf.cameras[
                            key].init_args.rosmaster_ip is not None else "localhost",
                        rosmaster_port=conf.cameras[key].init_args.rosmaster_port if conf.cameras[
                            key].init_args.rosmaster_port is not None else 9090
                    )
                case "RsRos":
                    self.__cam[key] = RsRos(
                        camera_node=conf.cameras[key].init_args.camera_node,
                        camera_type=conf.cameras[key].init_args.camera_type,
                        rosmaster_ip=conf.cameras[key].init_args.rosmaster_ip if conf.cameras[
                            key].init_args.rosmaster_ip is not None else "localhost",
                        rosmaster_port=conf.cameras[key].init_args.rosmaster_port if conf.cameras[
                            key].init_args.rosmaster_port is not None else 9090
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

    def robot_move_to_pose(self, pose: np.ndarray) -> None:
        for _ in range(3):
            while not self.__robot.move_to_pose(
                position=pose[:3, 3], orientation=pose[:3, :3]
            ):
                print(f"Failed to move to pose.")

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
