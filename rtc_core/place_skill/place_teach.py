import os
import time

import cv2
import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

from rtc_core.devices.devices import Devices


class TeachPlace:

    def __init__(self, config: DictConfig):

        self.devices = Devices(config.devices)
        self.config = config
        self.data_dir: str = None
        self.object: str = None
        self.debug = config.debug
        self.num_demos: int = None
        self.current_demo: int = None
        self.object = config.object

        current_dir = os.path.dirname(os.path.realpath(__file__))
        project_dir = os.path.join(current_dir, "../..")
        self.data_dir = os.path.join(project_dir, config.data_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.data_dir = config.data_dir
        self.debug = config.debug
        self.num_demos = config.num_demos
        self.current_demo = 0

        self.cam_keys = list(config.devices.cameras.keys())
        self.cam_setup = {key: {} for key in self.cam_keys}
        for key in self.cam_setup.keys():
            for subkey in config.devices.cameras[key]['setup'].keys():
                filepath = config.devices.cameras[key]['setup'][subkey]
                self.cam_setup[key][subkey] = np.load(filepath)
                
        # TODO: Remove this temporary fix
        self.ih_view_pose = None

    def collect_demonstrations(self) -> None:
        """
        Collects demonstration data for object placement.
        """

        print(f"COLLECTING PLACE DEMONSTRATION DATA FOR {self.object.upper()}")

        print("####################################################################")
        print("SETUP: 1. GO TO HOME POSE")
        print("####################################################################")
        
        home_pose = self.teach_pose("home_pose")
        
        print("####################################################################")
        print("SETUP: 2. GO TO A PLACE POSE")
        print("####################################################################")

        print("Move to placement pose.")
        input("Press Enter to close gripper and continue...")

        placement_pose = self.devices.robot_get_eef_pose()

        time.sleep(0.5)
        
        print("####################################################################")
        print("SETUP: 3. PULL UP AND INSERT AGAIN")
        print("####################################################################")

        print("Closing gripper")
        self.devices.gripper_close()
        time.sleep(0.5)

        print("Pulling up action object...")
        pre_placement_pose = np.copy(placement_pose)
        pre_placement_pose[2, 3] = pre_placement_pose[2,3] + self.config.training.target.pull_distance
        self.devices.robot_move_to_pose(pre_placement_pose)
        
        print("Insert the object in hand.")
        input("Press Enter when done...")        
        self.devices.gripper_open()
        
        print("####################################################################")
        print("SETUP DONE")
        print("####################################################################")        

        while self.current_demo < self.num_demos:
            
            print(f"STARTING DEMO {self.current_demo} of {self.num_demos}")

            print("####################################################################")
            print(f"DEMO {self.current_demo}: 1. Pre-grasp action object@target")
            print("####################################################################")

            print("Move to placement pose.")
            input("Press Enter to close gripper and continue...")

            placement_pose = self.devices.robot_get_eef_pose()

            time.sleep(0.5)

            print("####################################################################")
            print(f"DEMO {self.current_demo}: 2. Placement pose (grasp and pull)")
            print("####################################################################")

            print("Closing gripper")
            self.devices.gripper_close()
            time.sleep(0.5)

            print("Pulling up action object...")
            pre_placement_pose = np.copy(placement_pose)
            pre_placement_pose[2, 3] = pre_placement_pose[2,3] + self.config.training.target.pull_distance
            self.devices.robot_move_to_pose(pre_placement_pose)
            
            print("####################################################################")
            print(f"DEMO {self.current_demo}: 3. In-hand camera view")
            print("####################################################################")

            T_eef2camera = self.cam_setup[self.config.training.anchor.camera]["T_eef2cam"]
            T_eef2camera[2, 3] = 0.0
            ih_camera_view_pose = np.dot(
                pre_placement_pose, np.linalg.inv(T_eef2camera))
            
            if 'viewing_distance' in self.config.training.anchor.keys():
                ih_camera_view_pose[2, 3] = placement_pose[2, 3] \
                                            + self.config.training.anchor.viewing_distance \
                                            - 0.212 # 0.212 is the distance from the flange to the end effector tip          
            
            # TODO: Remove this temporary fix
            if self.ih_view_pose is not None:
                ih_camera_view_pose = self.ih_view_pose
                print("Using previous in-hand camera view pose")
            #check if ih_camera_view_pose.npy file exists in data directory
            elif os.path.exists(os.path.join(self.data_dir, "poses", "ih_camera_view_pose.npy")):
                ih_camera_view_pose = np.load(os.path.join(self.data_dir, "poses", "ih_camera_view_pose.npy"))
            else:
                self.ih_view_pose = ih_camera_view_pose         
                print("Using new in-hand camera view pose")       
            
            self.devices.robot_move_to_pose(ih_camera_view_pose)
            self.collect_data("ih_camera_view0")
            
            current_pose = ih_camera_view_pose
            if "view_variations" in self.config.training.anchor.keys():
                cfg = self.config.training.anchor.view_variations
                for i in range(cfg['count'] - 1):
                    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                    print(f"Capturing view variation {i+1} for in-hand camera view")
                    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                    delta_position = np.random.uniform(
                        -cfg['position'], cfg['position'], size=(3,))
                    delta_euler = np.random.uniform(
                        -cfg['orientation'], cfg['orientation'], size=(3,))
                    
                    print(f"deltas: {delta_position}, {delta_euler}")
                    
                    delta_pose = np.eye(4)
                    delta_pose[:3, :3] = R.from_euler('xyz', delta_euler, degrees=True).as_matrix()
                    delta_pose[:, 3] = np.array([*delta_position, 1.0])
                    pose = np.dot(current_pose, delta_pose)

                    if pose[2, 3] < current_pose[2, 3]:
                        pose[2, 3] = current_pose[2, 3]

                    if self.devices.robot_move_to_pose(pose):
                       self.collect_data(f"ih_camera_view{i+1}")
                    else:
                        i -= 1

            print("####################################################################")
            print(f"DEMO {self.current_demo}: 4. Gripper close-up view")
            print("####################################################################")

            self.devices.robot_move_to_pose(home_pose)

            print(f"Moving to object in hand close up pose...")
            T_base2camera = self.cam_setup[self.config.training.action.camera]["T_base2cam"]

            distance = self.cfg.execution.action.viewing_distance
            T_camera2gripper = np.asarray(self.cfg.devices.gripper.T_camera2gripper)
            T_eef2gripper = np.asarray(self.devices.gripper.T_ee2gripper)
            T_base2gripper = (T_base2camera @ T_camera2gripper) @ np.linalg.inv(T_eef2gripper)
            gripper_close_up_pose = T_base2gripper

            self.devices.robot_move_to_pose(gripper_close_up_pose)
            self.collect_data("gripper_close_up_view0")
            # breakpoint()
            current_pose = T_camera2gripper
            if "view_variations" in self.config.training.action.keys():
                cfg = self.config.training.action.view_variations
                for i in range(cfg['count'] - 1):
                    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                    print(f"Capturing view variation {i+1} for grip close-up view")
                    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                    delta_position = np.random.uniform(
                        -cfg['position'], cfg['position'], size=(3,))
                    delta_euler = np.random.uniform(
                        -cfg['orientation'], cfg['orientation'], size=(3,))
                    
                    print(f"deltas: {delta_position}, {delta_euler}")
                    
                    delta_pose = np.eye(4)
                    delta_pose[:3, :3] = R.from_euler('xyz', delta_euler, degrees=True).as_matrix()
                    delta_pose[:, 3] = np.array([*delta_position, 1.0])
                    pose = np.dot(current_pose, delta_pose)

                    if self.devices.robot_move_to_pose(np.dot(T_base2camera, pose)):
                       self.collect_data(f"gripper_close_up_view{i+1}")
                    else:
                        i -= 1

            print("####################################################################")
            print(f"DEMO {self.current_demo}: 5. Assemble it Back")
            print("####################################################################")

            self.devices.robot_move_to_pose(home_pose)
            print(f"Moving to placement pose...")
            self.devices.robot_move_to_pose(pre_placement_pose)

            is_good = input("Is the gripper centered? Press 'n' to place manually (y/n): ")
            if is_good == "n":
                input("Jog robot to placement pose and press Enter to continue...")
                self.current_demo -= 1
            elif is_good == "y":
                self.devices.robot_move_to_pose(placement_pose)
            else:
                print("Invalid input. Assuming 'n'...")
                input("Jog robot to placement pose and press Enter to continue...")
            
            self.collect_data("placement")
            
            self.devices.gripper_open()

            print("####################################################################")
            print(f"DEMO {self.current_demo}: DONE")
            print("####################################################################")
            
            self.current_demo += 1

        self.devices.robot_move_to_pose(home_pose)
        print("####################################################################")
        print(f"DATA COLLECTION for {self.object.upper()} DONE")
        print("####################################################################")

    def teach_pose(self, pose_name: str) -> None:
        """
        Teach a pose by name.

        Args:
            pose_name (str): The name of the pose to teach.

        Returns:
            None
        """

        poses_folder = os.path.join(self.data_dir, "poses")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)

        file_path = os.path.join(poses_folder, f"{pose_name}.npy")
        if not os.path.exists(file_path):
            print(f"No {pose_name} found. Show me!")
            input(f"Move robot to {pose_name} and press Enter to save...")
            pose = self.devices.robot_get_eef_pose()

            if pose_name not in ["placement_pose", "pre_placement_pose"]:
                np.save(file_path, pose)
        else:
            print(f"{pose_name} found. Reading and moving to pose...")
            pose = np.load(file_path)
            self.devices.robot_move_to_pose(pose)

            char = input(
                f"Press Enter to continue, and 'n' to modify {pose_name}..."
            )
            if char == "n":
                input(f"Move robot to {pose_name} and press Enter to save...")
                pose = self.devices.robot_get_eef_pose()
                np.save(file_path, pose)

        return pose

    def collect_data(self, robot_state: str) -> None:
        """
        Collects data from cameras and the robot for a given robot state.
        """
        
        data_dir = os.path.join(self.data_dir, "teach_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)        
        
        max_depth = 1500
        if "gripper_close_up" in robot_state or "ih_camera_view" in robot_state:
            max_depth = 305  # 254 mm = 10 inches, 305 mm = 12 inches

            # Collect data from cameras
            for cam_name in self.cam_keys:
                print(f"Collecting data from {cam_name}...")
                image = self.devices.cam_get_rgb_image(cam_name)
                depth_data = self.devices.cam_get_raw_depth_data(
                    cam_name, max_depth=max_depth)
                depth_image = self.devices.cam_get_colormap_depth_image(
                    cam_name, max_depth=max_depth)
                point_cloud = self.devices.cam_get_point_cloud(
                    cam_name, max_mm=max_depth)

                # save images in img folder
                img_folder = os.path.join(data_dir, "img_data")
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                cv2.imwrite(
                    os.path.join(
                        img_folder, f"demo{self.current_demo}_{robot_state}_{cam_name}_rgb.png"), image
                )
                cv2.imwrite(
                    os.path.join(
                        img_folder, f"demo{self.current_demo}_{robot_state}_{cam_name}_depth.png"),
                    depth_image,
                )
                np.save(
                    os.path.join(
                        img_folder, f"demo{self.current_demo}_{robot_state}_{cam_name}_depth_data.npy"),
                    depth_data,
                )

                # save point cloud in pcd folder
                pcd_folder = os.path.join(data_dir, "pcd_data")
                if not os.path.exists(pcd_folder):
                    os.makedirs(pcd_folder)
                o3d.io.write_point_cloud(
                    os.path.join(
                        pcd_folder, f"demo{self.current_demo}_{robot_state}_{cam_name}_pointcloud.ply"),
                    point_cloud,
                )

        # Collect data from robot
        eef_pose = self.devices.robot_get_eef_pose()
        poses_folder = os.path.join(data_dir, "pose_data")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)
        np.save(os.path.join(poses_folder,
                f"demo{self.current_demo}_{robot_state}_pose.npy"), eef_pose)
