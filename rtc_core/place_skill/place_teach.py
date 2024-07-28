import os
import time

import cv2
import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf

from rtc_core.devices.devices import Devices


class TeachPlace:

    def __init__(self, config: DictConfig):        

        self.__devices = Devices(config.devices)
        self.__data_dir: str = None
        self.__object: str = None
        self.__debug = config.debug
        self.__num_demos: int = None
        self.__current_demo: int = None
        self.__object = config.object
        self.__poses = {
            "start_pose": None,
            "out_of_way_pose": None,
            "gripper_close_up_pose": None,
            "placement_pose": None,
            "pre_placement_pose": None,
            "ih_camera_view_pose": None,
        }
        self.__cam_keys = list(config.devices.cameras.keys())
        self.__cam_setup = {key: {} for key in self.__cam_keys}
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        project_dir = os.path.join(current_dir, "../..")        
        self.__data_dir = os.path.join(project_dir, config.data_dir)
        if not os.path.exists(self.__data_dir):
            os.makedirs(self.__data_dir)
            
        self.__data_dir = config.data_dir
        self.__debug = config.debug
        self.__num_demos = config.num_demos
        self.__current_demo = 0
        
        for key in self.__cam_setup.keys():
            if key not in config.devices.cameras.keys():
                continue
            for subkey in config.devices.cameras[key]['setup'].keys():
                filepath = config.devices.cameras[key]['setup'][subkey]
                self.__cam_setup[key][subkey] = np.load(filepath)

    def collect_demonstration(self) -> None:
        """
        Collects demonstration data for object placement.

        Raises:
            Exception: If setup is not done. Call setup() before calling teach_pose().
        """

        print(f"COLLECTING PLACE DEMONSTRATION DATA FOR {self.__object.upper()}")

        while self.__current_demo <= self.__num_demos:
            self.__current_demo += 1
            print(f"\nSTART COLLECTION FOR DEMO {self.__current_demo} of {self.__num_demos}")

            print("################################################################################")
            print("Start Pose")
            print("################################################################################")
            self.__teach_pose("start_pose")

            time.sleep(0.5)

            print("################################################################################")
            print("Grasp Object")
            print("################################################################################")

            grasp_again = input("Repeat Grasp? (y/n): ")
            
            if grasp_again == "y":            
                print("Open gripper")
                self.__devices.gripper_open()
                input("Press Enter to close gripper when object inside gripper...")
                self.__devices.gripper_close()

            time.sleep(0.5)

            print("################################################################################")
            print("Out of Way Pose")
            print("################################################################################")

            self.__teach_pose("out_of_way_pose")

            time.sleep(0.5)

            self.__collect_data("out_of_way")

            print("################################################################################")
            print("Gripper Close Up Pose")
            print("################################################################################")

            self.__teach_pose("start_pose")

            print(f"Moving to object in hand close up pose...")
            self.__teach_pose("gripper_close_up_pose")

            time.sleep(0.5)

            self.__collect_data("gripper_close_up")

            print("################################################################################")
            print("Placement Pose")
            print("################################################################################")

            time.sleep(0.5)

            print(f"Moving to placement pose...")
            self.__teach_pose("placement_pose")

            self.__collect_data("placement")

            print("################################################################################")
            print("Pre Placement Pose")
            print("################################################################################")

            print(f"Moving to pre placement pose...")
            self.__teach_pose("pre_placement_pose")

            time.sleep(0.5)

            self.__collect_data("pre_placement")

            print("################################################################################")
            print("InHand Camera View")
            print("################################################################################")

            T_eef2camera = self.__cam_setup["cam3_gripper"]["T_eef2cam"]
            T_eef2camera[2,3] = 0.0
            self.__poses['ih_camera_view_pose'] = np.dot(
                self.__poses['pre_placement_pose'], np.linalg.inv(T_eef2camera))
            self.__teach_pose("ih_camera_view_pose")

            self.__collect_data("ih_camera_view")

            print(f"END COLLECTION FOR DEMO {self.__current_demo}")
            self.__teach_pose("start_pose")

        print("Done collecting data. Moving to start pose...")
        self.__teach_pose("start_pose")

    def __teach_pose(self, pose_name: str) -> None:
        """
        Teach a pose by name.

        Args:
            pose_name (str): The name of the pose to teach.

        Returns:
            None
        """

        if self.__poses[pose_name] is not None and pose_name not in ["placement_pose", "pre_placement_pose"]:
            print(f"{pose_name} already taught. Moving to pose...")
            self.__devices.robot_move_to_pose(self.__poses[pose_name])
            return

        poses_folder = os.path.join(self.__data_dir, "poses")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)

        file_path = os.path.join(poses_folder, f"{pose_name}.npy")
        if not os.path.exists(file_path):
            print(f"No {pose_name} found. Show me!")
            input(f"Move robot to {pose_name} and press Enter to save...")
            pose = self.__devices.robot_get_eef_pose()

            if pose_name not in ["placement_pose", "pre_placement_pose"]:
                np.save(file_path, pose)

        else:
            print(f"{pose_name} found. Reading and moving to pose...")
            pose = np.load(file_path)
            self.__devices.robot_move_to_pose(pose)

            char = input(
                f"Press Enter to continue, and 'n' to modify {pose_name}..."
            )
            if char == "n":
                input(f"Move robot to {pose_name} and press Enter to save...")
                pose = self.__devices.robot_get_eef_pose()
                np.save(file_path, pose)
        self.__poses[pose_name] = pose

    def __collect_data(self, robot_state: str) -> None:
        """
        Collects data from cameras and the robot for a given robot state.

        Args:
            robot_state (str): The current state of the robot.

        Returns:
            None
        """

        char = input(f"Press Enter to collect data for {robot_state}...")
        if char == "n":
            return

        max_depth = 1500
        if robot_state in ["gripper_close_up", "ih_camera_view_pose"]:
            max_depth = 305  # 254 mm = 10 inches, 305 mm = 12 inches

        data_dir = os.path.join(self.__data_dir, "teach_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Collect data from cameras
        for cam_name in self.__cam_keys:
            print(f"Collecting data from {cam_name}...")
            image = self.__devices.cam_get_rgb_image(cam_name)
            depth_data = self.__devices.cam_get_raw_depth_data(
                cam_name, max_depth=max_depth)
            depth_image = self.__devices.cam_get_colormap_depth_image(
                cam_name, max_depth=max_depth)
            point_cloud = self.__devices.cam_get_point_cloud(cam_name, max_mm=max_depth)

            # save images in img folder
            img_folder = os.path.join(data_dir, "img_data")
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            cv2.imwrite(
                os.path.join(
                    img_folder, f"demo{self.__current_demo}_{robot_state}_{cam_name}_rgb.png"), image
            )
            cv2.imwrite(
                os.path.join(
                    img_folder, f"demo{self.__current_demo}_{robot_state}_{cam_name}_depth.png"),
                depth_image,
            )
            np.save(
                os.path.join(
                    img_folder, f"demo{self.__current_demo}_{robot_state}_{cam_name}_depth_data.npy"),
                depth_data,
            )

            # save point cloud in pcd folder
            pcd_folder = os.path.join(data_dir, "pcd_data")
            if not os.path.exists(pcd_folder):
                os.makedirs(pcd_folder)
            o3d.io.write_point_cloud(
                os.path.join(
                    pcd_folder, f"demo{self.__current_demo}_{robot_state}_{cam_name}_pointcloud.ply"),
                point_cloud,
            )

        # Collect data from robot
        eef_pose = self.__devices.robot_get_eef_pose()
        poses_folder = os.path.join(data_dir, "pose_data")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)
        np.save(os.path.join(poses_folder,
                f"demo{self.__current_demo}_{robot_state}_pose.npy"), eef_pose)
