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
from rtc_core.place_skill.place_teach import TeachPlace


class TeachPlace_:
    """
    A class for setting up and collecting demonstration data for object placement. 
    
    The process follows the following steps for each demonstration:
    Start Pose -> Grasp Object -> Out of Way Pose -> Gripper Close Up Pose 
    -> Placement Pose -> Pre Placement Pose -> InHand Camera View

    Attributes:
        __poses (dict): A dictionary of pose names and their corresponding values.
        __cam_obj (dict): A dictionary of camera names and their corresponding camera objects.
        __robot: The robot object.
        __gripper: The gripper object.
        __data_dir (str): The directory path for storing data.
        __object (str): The name of the object.
        __debug (bool): A flag indicating whether debug mode is enabled.
        __setup_done (bool): A flag indicating whether the setup has been done.
        __num_demos (int): The number of demonstrations to collect.
        __current_demo (int): The index of the current demonstration.

    Methods:
        setup(config: DictConfig, object: str) -> None:
            Sets up the place teaching environment.

        collect_demonstration() -> None:
            Collects demonstration data for object placement.

        __teach_pose(pose_name: str) -> None:
            Teach a pose by name.

        __collect_data(robot_state: str) -> None:
            Collects data from cameras and the robot for a given robot state.
    """

    __poses = {
        "start_pose": None,
        "out_of_way_pose": None,
        "gripper_close_up_pose": None,
        "placement_pose": None,
        "pre_placement_pose": None,
        "ih_camera_view_pose": None,
    }
    __cam_obj = {
        "cam0_board": None,
        "cam1_board": None,
        "cam2_closeup": None,
        "cam3_gripper": None,
    }
    __cam_setup = {
        "cam0_board": {},
        "cam1_board": {},
        "cam2_closeup": {},
        "cam3_gripper": {},
    }
    __robot = None
    __gripper = None
    __data_dir: str = None
    __object: str = None
    __debug = False
    __setup_done = False
    __num_demos: int = None
    __current_demo: int = None

    @classmethod
    def setup(cls, config: DictConfig):
        """
        Sets up the place teaching environment.

        Args:
            config_file (str): The path to the configuration file.
            object (str): The name of the object.

        Raises:
            Exception: If the object name is not provided.

        Returns:
            None
        """
        if object is None:
            raise Exception("Object not provided. Please provide object name.")

        cls.__object = config.object

        # setup class variables using config
        cls.__initialize_devices(config)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        project_dir = os.path.join(current_dir, "..")
        
        cls.__data_dir = os.path.join(project_dir, config.data_dir)
        if not os.path.exists(cls.__data_dir):
            os.makedirs(cls.__data_dir)
        cls.__debug = config.debug
        cls.__num_demos = config.num_demos
        cls.__setup_done = True
        cls.__current_demo = 0
        
        for key in cls.__cam_setup.keys():
            if key not in config.keys():
                continue
            for subkey in config[key]['setup'].keys():
                filepath = os.path.join(project_dir, config[key]['setup'][subkey])
                cls.__cam_setup[key][subkey] = np.load(filepath)

    @classmethod
    def __initialize_devices(cls, conf: DictConfig) -> None:
        """
        Initializes the camera and robot devices.

        Args:
            conf (DictConfig): The configuration object.

        Returns:
            None
        """
        # Initialize cameras
        for key in cls.__cam_obj.keys():
            if key not in conf.keys():
                continue
            match conf[key].type:
                case "ZedRos":
                    cls.__cam_obj[key] = ZedRos(
                        camera_node=conf[key].init_args.camera_node,
                        camera_type=conf[key].init_args.camera_type,
                    )
                case "RsRos":
                    cls.__cam_obj[key] = RsRos(
                        camera_node=conf[key].init_args.camera_node,
                        camera_type=conf[key].init_args.camera_type,
                    )
                case "OBCamera":
                    cls.__cam_obj[key] = OBCamera(
                        serial_no=conf[key].init_args.serial_number
                    )

        # Initialize robot
        cls.__robot = ROSRobot(
            robot_name=conf.robot.init_args.robot_name,
            rosmaster_ip=conf.robot.init_args.rosmaster_ip,
        )

        # Initialize gripper
        cls.__gripper = RobotiqGripper(portname=conf.gripper.init_args.port)

    @classmethod
    def collect_demonstration(cls) -> None:
        """
        Collects demonstration data for object placement.

        Raises:
            Exception: If setup is not done. Call setup() before calling teach_pose().
        """
        if cls.__setup_done is False:
            raise Exception(
                "Setup not done. Call setup() before calling teach_pose()")

        print(f"COLLECTING PLACE DEMONSTRATION DATA FOR {cls.__object.upper()}")

        while cls.__current_demo <= cls.__num_demos:
            cls.__current_demo += 1
            print(f"\nSTART COLLECTION FOR DEMO {cls.__current_demo} of {cls.__num_demos}")

            print("################################################################################")
            print("Start Pose")
            print("################################################################################")
            cls.__teach_pose("start_pose")

            time.sleep(0.5)

            print("################################################################################")
            print("Grasp Object")
            print("################################################################################")

            grasp_again = input("Repeat Grasp? (y/n): ")
            
            if grasp_again == "y":            
                print("Open gripper")
                cls.__gripper_open()
                input("Press Enter to close gripper when object inside gripper...")
                cls.__gripper_close()

            time.sleep(0.5)

            print("################################################################################")
            print("Out of Way Pose")
            print("################################################################################")

            cls.__teach_pose("out_of_way_pose")

            time.sleep(0.5)

            cls.__collect_data("out_of_way")

            print("################################################################################")
            print("Gripper Close Up Pose")
            print("################################################################################")

            cls.__teach_pose("start_pose")

            print(f"Moving to object in hand close up pose...")
            cls.__teach_pose("gripper_close_up_pose")

            time.sleep(0.5)

            cls.__collect_data("gripper_close_up")

            print("################################################################################")
            print("Placement Pose")
            print("################################################################################")

            time.sleep(0.5)

            print(f"Moving to placement pose...")
            cls.__teach_pose("placement_pose")

            cls.__collect_data("placement")

            print("################################################################################")
            print("Pre Placement Pose")
            print("################################################################################")

            print(f"Moving to pre placement pose...")
            cls.__teach_pose("pre_placement_pose")

            time.sleep(0.5)

            cls.__collect_data("pre_placement")

            print("################################################################################")
            print("InHand Camera View")
            print("################################################################################")

            T_eef2camera = cls.__cam_setup["cam3_gripper"]["T_eef2cam"]
            T_eef2camera[2,3] = 0.0
            cls.__poses['ih_camera_view_pose'] = np.dot(
                cls.__poses['pre_placement_pose'], np.linalg.inv(T_eef2camera))
            cls.__teach_pose("ih_camera_view_pose")

            cls.__collect_data("ih_camera_view")

            print(f"END COLLECTION FOR DEMO {cls.__current_demo}")
            cls.__teach_pose("start_pose")

        print("Done collecting data. Moving to start pose...")
        cls.__teach_pose("start_pose")

    @classmethod
    def __teach_pose(cls, pose_name: str) -> None:
        """
        Teach a pose by name.

        Args:
            pose_name (str): The name of the pose to teach.

        Returns:
            None
        """

        if cls.__poses[pose_name] is not None and pose_name not in ["placement_pose", "pre_placement_pose"]:
            print(f"{pose_name} already taught. Moving to pose...")
            cls.__robot_move_to_pose(cls.__poses[pose_name])
            return

        poses_folder = os.path.join(cls.__data_dir, "poses")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)

        file_path = os.path.join(poses_folder, f"{pose_name}.npy")
        if not os.path.exists(file_path):
            print(f"No {pose_name} found. Show me!")
            input(f"Move robot to {pose_name} and press Enter to save...")
            pose = cls.__robot_get_eef_pose()

            if pose_name not in ["placement_pose", "pre_placement_pose"]:
                np.save(file_path, pose)

        else:
            print(f"{pose_name} found. Reading and moving to pose...")
            pose = np.load(file_path)
            cls.__robot_move_to_pose(pose)

            char = input(
                f"Press Enter to continue, and 'n' to modify {pose_name}..."
            )
            if char == "n":
                input(f"Move robot to {pose_name} and press Enter to save...")
                pose = cls.__robot_get_eef_pose()
                np.save(file_path, pose)
        cls.__poses[pose_name] = pose

    @classmethod
    def __collect_data(cls, robot_state: str) -> None:
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

        data_dir = os.path.join(cls.__data_dir, "teach_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Collect data from cameras
        for cam_name in cls.__cam_obj.keys():
            if cls.__cam_obj[cam_name] is None:
                continue
            print(f"Collecting data from {cam_name}...")
            image = cls.__cam_get_rgb_image(cam_name)
            depth_data = cls.__cam_get_raw_depth_data(
                cam_name, max_depth=max_depth)
            depth_image = cls.__cam_get_colormap_depth_image(
                cam_name, max_depth=max_depth)
            point_cloud = cls.__cam_get_point_cloud(cam_name, max_mm=max_depth)

            # save images in img folder
            img_folder = os.path.join(data_dir, "img_data")
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            cv2.imwrite(
                os.path.join(
                    img_folder, f"demo{cls.__current_demo}_{robot_state}_{cam_name}_rgb.png"), image
            )
            cv2.imwrite(
                os.path.join(
                    img_folder, f"demo{cls.__current_demo}_{robot_state}_{cam_name}_depth.png"),
                depth_image,
            )
            np.save(
                os.path.join(
                    img_folder, f"demo{cls.__current_demo}_{robot_state}_{cam_name}_depth_data.npy"),
                depth_data,
            )

            # save point cloud in pcd folder
            pcd_folder = os.path.join(data_dir, "pcd_data")
            if not os.path.exists(pcd_folder):
                os.makedirs(pcd_folder)
            o3d.io.write_point_cloud(
                os.path.join(
                    pcd_folder, f"demo{cls.__current_demo}_{robot_state}_{cam_name}_pointcloud.ply"),
                point_cloud,
            )

        # Collect data from robot
        eef_pose = cls.__robot_get_eef_pose()
        poses_folder = os.path.join(data_dir, "pose_data")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)
        np.save(os.path.join(poses_folder,
                f"demo{cls.__current_demo}_{robot_state}_pose.npy"), eef_pose)

    @classmethod
    def __robot_get_eef_pose(cls) -> np.ndarray:
        return cls.__robot.get_eef_pose()

    @classmethod
    def __robot_move_to_pose(cls, pose: np.ndarray) -> None:
        for _ in range(3):
            while not cls.__robot.move_to_pose(
                position=pose[:3, 3], orientation=pose[:3, :3]
            ):
                print(f"Failed to move to pose.")

    @classmethod
    def __gripper_open(cls) -> None:
        cls.__gripper.openGripper()

    @classmethod
    def __gripper_close(cls) -> None:
        cls.__gripper.closeGripper()

    @classmethod
    def __cam_get_rgb_image(cls, cam_name: str) -> np.ndarray:
        return cls.__cam_obj[cam_name].get_rgb_image()

    @classmethod
    def __cam_get_raw_depth_data(cls, cam_name: str, max_depth: int) -> np.ndarray:
        if max_depth is None:
            return cls.__cam_obj[cam_name].get_raw_depth_data()
        else:
            return cls.__cam_obj[cam_name].get_raw_depth_data(max_depth=max_depth)

    @classmethod
    def __cam_get_colormap_depth_image(cls, cam_name: str, max_depth: int) -> np.ndarray:
        if max_depth is None:
            return cls.__cam_obj[cam_name].get_colormap_depth_image()
        else:
            return cls.__cam_obj[cam_name].get_colormap_depth_image(max_depth=max_depth)

    @classmethod
    def __cam_get_point_cloud(cls, cam_name: str, max_mm: int) -> o3d.geometry.PointCloud:
        if max_mm is None:
            return cls.__cam_obj[cam_name].get_point_cloud()
        else:
            return cls.__cam_obj[cam_name].get_point_cloud(max_mm=max_mm)


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
    
    # TeachPlace.setup(config)
    # TeachPlace.collect_demonstration()
    
    place_teach = TeachPlace(config)
    place_teach.collect_demonstration()
