import argparse
import datetime
import glob
import os
import time
from typing import Dict, List, Tuple

import cv2
import hydra
import numpy as np
import open3d as o3d
import torch
from omegaconf import DictConfig
from pytorch3d.transforms import Transform3d
from taxpose.datasets.augmentations import maybe_downsample
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import \
    EquivarianceTrainingModule
from taxpose.utils.load_model import get_weights_path
from taxpose.utils.se3 import random_se3

from camera.orbbec.ob_camera import OBCamera
from camera.rs_ros.rs_ros import RsRos
from camera.zed_ros.zed_ros import ZedRos
from robot.robotiq.robotiq_gripper import RobotiqGripper
from robot.ros_robot.ros_robot import ROSRobot

import copy

CONFIG_DIR = "/home/mfi/repos/rtc_vision_toolbox/config/"
DATA_DIR = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/"
# DATA_DIR = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-27-dsub/"

INHAND_DEVICES_SN = ["CL8FC3100NM"]
BOARD_DEVICES_SN = ["CL8FC3100RL", "CL8FC3100W3"]
ROBOT_IP = "172.26.179.142"
ROBOT_NAME = "yk_builder"
GRIPPER_PORT = "/dev/ttyUSB0"

T_eef2camera_filepath = "/home/mfi/repos/rtc_vision_toolbox/data/calibration_data/rs/T_eef2camera.npy"
T_base2cam2_filepath = "/home/mfi/repos/rtc_vision_toolbox/data/calibration_data/zed/T_base2cam2.npy"

class ExecutePlace:

    __poses = {
        "start_pose": None,
        "out_of_way_pose": None,
        "gripper_close_up_pose": None,
        "pre_placement_offset": None,
    }
    __cam_obj = {
        "cam0_board": None,
        "cam1_board": None,
        "cam2_close_up": None,
        "cam3_gripper": None,
    }
    cam_setup = {
        "cam0_board": {},
        "cam1_board": {},
        "cam2_close_up": {},
        "cam3_gripper": {},
    }
    __predict_board_pose_data = {
        "image_segmentation_model": None,
        "out_of_way": {
            "cam0": {
                "rgb": None,
                "depth_data": None,
            }
        }
    }
    predict_placement_pose_data = {
        "taxpose_model": None,
        "ih_camera_view": {
            "cam3_gripper": {
                "pcd": None,
            },
            "eef_pose": None,
        },
        "gripper_close_up": {
            "cam2_close_up": {
                "pcd": None,
            },
            "eef_pose": None,
        }
    }

    __robot = None
    __gripper = None
    __data_dir: str = None
    __save_dir: str = None
    __object: str = None
    __debug = False
    __setup_done = False

    @classmethod
    def setup(cls, cfg: DictConfig, object: str):
        if object is None:
            raise Exception("Object not provided. Please provide object name.")

        cls.__object = object

        '''
        cam0_board (type, init args, setup args)
        cam1_board (type, init args, setup args)
        cam2_closeup (type, init args, setup args)
        cam3_gripper (type, init args, setup args)
        robot (type, init args, setup args)
        gripper (type, init args, setup args)
        debug
        data_dir
        
        image_segmentation (type, init args, setup args)
        object_pose_estimation (type, init args, setup args)
        '''
        # setup class variables using config
        cls.__data_dir = DATA_DIR
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        cls.__save_dir = os.path.join(DATA_DIR, f"execute_data/{now}")
        
        if not os.path.exists(cls.__data_dir):
            raise Exception(f"Data directory {cls.__data_dir} does not exist.")
        else:
            print(f"Data directory: {cls.__data_dir}")
            
        if not os.path.exists(cls.__save_dir):
            os.makedirs(cls.__save_dir)
            print(f"Save directory: {cls.__save_dir}")        
        
        cls.__debug = False
        cls.__setup_done = True
        
        cls.predict_placement_pose_data['taxpose_model'] = cls.load_taxpose_model(cfg)

        # cls.__robot = ROSRobot(robot_name=ROBOT_NAME, rosmaster_ip=ROBOT_IP)
        # cls.__gripper = RobotiqGripper(GRIPPER_PORT)

        # cls.__cam_obj = {
        #     "cam0_board": ZedRos(camera_node="/cam0/zed_cam0", camera_type="zedx"),
        #     "cam2_close_up": ZedRos(camera_node="/cam2/zed_cam2", camera_type="zedxm"),
        #     "cam3_gripper": RsRos(camera_node="/camera/realsense2_camera", camera_type="d405"),
        # }
        cls.cam_setup = {
            "cam0_board": {
                "T_base2cam": None,
            },
            "cam1_board": {
                "T_base2cam": None,
            },
            "cam2_close_up": {
                "T_base2cam": np.load(T_base2cam2_filepath),
            },
            "cam3_gripper": {
                "T_eef2camera": np.load(T_eef2camera_filepath),
            },
        }
        cls.__poses = {
            "start_pose": np.load(os.path.join(cls.__data_dir, "poses", "start_pose.npy")),
            "out_of_way_pose": np.load(os.path.join(cls.__data_dir, "poses", "out_of_way_pose.npy")),
            "gripper_close_up_pose": np.load(os.path.join(cls.__data_dir, "poses", "gripper_close_up_pose.npy")),
            "pre_placement_offset": np.load(os.path.join(cls.__data_dir, "learn_data", "pre_placement_offset.npy")),
        }
        
        cls.__predict_board_pose_data['image_segmentation_model'] = None

    @classmethod
    def execute(cls) -> None:
        if cls.__setup_done is False:
            raise Exception(
                "Setup not done. Call setup() before calling teach_pose()")

        print(f"EXECUTING PLACE FOR {cls.__object.upper()}")

        print("################################################################################")
        print("1. Start Pose")
        print("################################################################################")

        cls.__robot_move_to_pose(cls.__poses['start_pose'])
        time.sleep(0.5)

        print("################################################################################")
        print("2. Grasp Object")
        print("################################################################################")

        print("Open gripper")

        cls.__gripper_open()

        time.sleep(0.5)

        input("Press Enter to close gripper when object inside gripper...")

        cls.__gripper_close()
        time.sleep(0.5)

        print("################################################################################")
        print("3. Out of Way Pose")
        print("################################################################################")

        cls.__robot_move_to_pose(cls.__poses['out_of_way_pose'])

        time.sleep(0.5)

        cls.__collect_data("out_of_way")
        time.sleep(0.5)

        print("################################################################################")
        print("4. Gripper Close Up Pose")
        print("################################################################################")

        cls.__robot_move_to_pose(cls.__poses['start_pose'])

        print(f"Moving to object in hand close up pose...")
        cls.__robot_move_to_pose(cls.__poses['gripper_close_up_pose'])

        time.sleep(0.5)

        cls.__collect_data("gripper_close_up")
        time.sleep(0.5)

        print("################################################################################")
        print("5. Predict and Move to Approx Pre Placement Pose")
        print("################################################################################")
        approx_pre_placement_pose = cls.infer_board_pose()

        print(f"Approx Pre Placement Pose: \n{approx_pre_placement_pose}")
        continue_execution = input("Press Enter to continue execution. Press 'c' to debug:")
        if continue_execution == "c":
            breakpoint()

        cls.__robot_move_to_pose(approx_pre_placement_pose)
        cls.__collect_data("ih_camera_view")
        
        time.sleep(0.5)

        print("################################################################################")
        print("7. Infer Placement Pose and Move to Placement Pose offset")
        print("################################################################################")

        placement_pose = cls.infer_placement_pose()
        offset_placement_pose = np.dot(placement_pose, cls.__poses['pre_placement_offset'])

        cls.__robot_move_to_pose(offset_placement_pose)
        time.sleep(0.5)

        print("################################################################################")
        print("8. Final Placement")
        print("################################################################################")

        continue_execution = input("Press Enter to continue execution. Press 'c' to debug:")
        if continue_execution == "c":
            breakpoint()

        cls.__robot_move_to_pose(placement_pose)
        time.sleep(0.5)

        print("################################################################################")
        print("9. Return to Start Pose")
        print("################################################################################")

        cls.__gripper_open()
        time.sleep(0.5)
        cls.__robot_move_to_pose(cls.__poses['start_pose'])

        print("################################################################################")
        print(f"\033[1m\033[3m{cls.__object.upper()} PLACEMENT DONE!\033[0m")

    @classmethod
    def __collect_data(cls, robot_state: str) -> None:
        """
        Collects data from cameras and the robot for a given robot state.

        Args:
            robot_state (str): The current state of the robot.

        Returns:
            None
        """

        # char = input(f"Press Enter to collect data for {robot_state}...")
        # if char == "n":
        #     return

        max_depth = 1500
        if robot_state in ["gripper_close_up", "ih_camera_view_pose"]:
            max_depth = 305  # 254 mm = 10 inches, 305 mm = 12 inches

        data_dir = cls.__save_dir

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
                    img_folder, f"{robot_state}_{cam_name}_rgb.png"), image
            )
            cv2.imwrite(
                os.path.join(
                    img_folder, f"{robot_state}_{cam_name}_depth.png"),
                depth_image,
            )
            np.save(
                os.path.join(
                    img_folder, f"{robot_state}_{cam_name}_depth_data.npy"),
                depth_data,
            )

            # save point cloud in pcd folder
            pcd_folder = os.path.join(data_dir, "pcd_data")
            if not os.path.exists(pcd_folder):
                os.makedirs(pcd_folder)
            o3d.io.write_point_cloud(
                os.path.join(
                    pcd_folder, f"{robot_state}_{cam_name}_pointcloud.ply"),
                point_cloud,
            )

            # get data for prediction
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^)")
            target_dict_list = [cls.__predict_board_pose_data,
                                cls.predict_placement_pose_data]
            for target_dict in target_dict_list:
                print("target_dict keys: ", target_dict.keys())
                for key in target_dict.keys():
                    if key == robot_state:
                        print("key: ", target_dict[key].keys())
                        for subkey in target_dict[key].keys():
                            if subkey == cam_name:
                                print("subkey: ", target_dict[key][subkey].keys())
                                for subsubkey in target_dict[key][subkey].keys():
                                    print(f"Saving {key}_{subkey}_{subsubkey}...")
                                    if subsubkey == "rgb":
                                        target_dict[key][subkey][subsubkey] = image
                                    elif subsubkey == "depth_data":
                                        target_dict[key][subkey][subsubkey] = depth_data
                                    elif subsubkey == "pcd":
                                        target_dict[key][subkey][subsubkey] = point_cloud
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^)")
            
        # Collect data from robot
        eef_pose = cls.__robot_get_eef_pose()
        poses_folder = os.path.join(data_dir, "pose_data")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)
        np.save(os.path.join(poses_folder,
                f"{robot_state}_pose.npy"), eef_pose)
        
        target_dict_list = [cls.__predict_board_pose_data,
                            cls.predict_placement_pose_data]
        for target_dict in target_dict_list:
            for key in target_dict.keys():
                if key == robot_state:
                    if subkey == 'eef_pose':
                        target_dict[key][subkey] = eef_pose
                        
    @classmethod
    def infer_board_pose(cls) -> np.ndarray:
        input("Move closer to the placement pose and Press Enter")
        return cls.__robot_get_eef_pose()
        
    @classmethod
    def infer_placement_pose(cls) -> np.ndarray:
        
        place_model = cls.predict_placement_pose_data['taxpose_model']
        
        action_pcd_raw = cls.predict_placement_pose_data['gripper_close_up']['cam2_close_up']['pcd']
        action_crop_box = {
                'min_bound': np.array([-200, -60, 0.0]),
                'max_bound': np.array([200, 60, 200]),
        }
        action_pcd = cls.__crop_point_cloud(action_pcd_raw, action_crop_box)
        action_pcd = cls.__filter_point_cloud(action_pcd)
        action_pcd.points = o3d.utility.Vector3dVector(np.asarray(action_pcd.points)/1000.0)
        
        anchor_pcd_raw = cls.predict_placement_pose_data['ih_camera_view']['cam3_gripper']['pcd']
        anchor_crop_box = {
                'min_bound': np.array([-53, -28, 0.0]),
                'max_bound': np.array([53, 25, 300])
        }
        anchor_pcd = cls.__crop_point_cloud(anchor_pcd_raw, anchor_crop_box)
        anchor_pcd = cls.__filter_point_cloud(anchor_pcd)
        anchor_pcd.points = o3d.utility.Vector3dVector(np.asarray(anchor_pcd.points)/1000.0)
        
        T_base2eef = cls.predict_placement_pose_data['ih_camera_view']['eef_pose']
        T_base2cam_anchor = np.dot(T_base2eef, cls.cam_setup['cam3_gripper']['T_eef2camera'])
        T_base2cam_action =  cls.cam_setup['cam2_close_up']['T_base2cam']
        
        print(f"anchor tf:\n {T_base2cam_anchor}")
        print(f"action tf:\n {T_base2cam_action}")        
        
        action_pcd_tf = action_pcd.transform(T_base2cam_action) 
        anchor_pcd_tf = anchor_pcd.transform(T_base2cam_anchor)
        
        action_points = np.asarray(action_pcd_tf.points).astype(np.float32)
        action_points = maybe_downsample(action_points[None, ...], 2048, 'fps')
        action_points = torch.from_numpy(action_points).to('cuda')
        
        anchor_points = np.asarray(anchor_pcd_tf.points).astype(np.float32)
        anchor_points = maybe_downsample(anchor_points[None, ...], 2048, 'fps')
        anchor_points = torch.from_numpy(anchor_points).cuda('cuda')
        
        print(f'action_points: {action_points[:, :3]}')
        print(f'anchor_points: {anchor_points[:, :3]}')

        place_out = place_model(action_points, anchor_points, None, None)
        predicted_place_rel_transform = place_out['pred_T_action']
        predicted_tf = predicted_place_rel_transform.get_matrix().T.squeeze(-1).detach().cpu().numpy()
        print(f'predicted_place_rel_transform: \n{predicted_place_rel_transform.get_matrix().T.squeeze(-1).detach().cpu().numpy()}')

        inhand_pose = cls.__poses['gripper_close_up_pose']
        inhand_pose_tf = Transform3d(
            matrix=torch.Tensor(inhand_pose.T), 
            device=predicted_place_rel_transform.device
        ).to(predicted_place_rel_transform.device)
        place_pose_tf = inhand_pose_tf.compose(predicted_place_rel_transform)
        place_pose = place_pose_tf.get_matrix().T.squeeze(-1).detach().cpu().numpy()
        
        # SAVE INFERENCE DATA
        predicted_tf = inhand_pose @ predicted_tf
        print(f"Predicted Pose: \n{np.round(predicted_tf,3)}")
        print(f"Predicted Placement Pose: \n{np.round(place_pose,3)}")
        print(f"InHand Pose: \n{np.round(inhand_pose,3)}")
        np.save(cls.__save_dir + "/predicted_pose.npy", place_pose)
        np.save(cls.__save_dir + "/predicted_tf.npy", predicted_tf)
        np.save(cls.__save_dir + "/inhand_pose.npy", inhand_pose)
        
        pred_place_points = predicted_place_rel_transform.transform_points(action_points)
        np.save(cls.__save_dir + "/pred_place_points.npy", pred_place_points[0].detach().cpu().numpy())
        np.save(cls.__save_dir + "/action_points.npy", action_points[0].detach().cpu().numpy())
        np.save(cls.__save_dir + "/anchor_points.npy", anchor_points[0].detach().cpu().numpy())
        
    
        return place_pose
   
    @classmethod
    def test_infer_placement_pose(cls, action_points, anchor_points) -> np.ndarray:
        
        place_model = cls.predict_placement_pose_data['taxpose_model']
        
        action_points = np.asarray(action_points).astype(np.float32)
        # action_points = maybe_downsample(action_points[None, ...], 2048, 'fps')
        action_points = maybe_downsample(action_points[None, ...], 2048, '2N_random_fps')
        action_points = torch.from_numpy(action_points).to('cuda')
        
        anchor_points = np.asarray(anchor_points).astype(np.float32)
        # anchor_points = maybe_downsample(anchor_points[None, ...], 2048, 'fps')
        anchor_points = maybe_downsample(anchor_points[None, ...], 2048, '2N_random_fps')
        anchor_points = torch.from_numpy(anchor_points).cuda('cuda')
        
        print(f'action_points: {action_points[:, :3]}')
        print(f'anchor_points: {anchor_points[:, :3]}')

        place_out = place_model(action_points, anchor_points, None, None)
        predicted_place_rel_transform = place_out['pred_T_action']
        predicted_tf = predicted_place_rel_transform.get_matrix().T.squeeze(-1).detach().cpu().numpy()
        print(f'predicted_place_rel_transform: \n{predicted_place_rel_transform.get_matrix().T.squeeze(-1).detach().cpu().numpy()}')

        inhand_pose = cls.__poses['gripper_close_up_pose']
        inhand_pose_tf = Transform3d(
            matrix=torch.Tensor(inhand_pose.T), 
            device=predicted_place_rel_transform.device
        ).to(predicted_place_rel_transform.device)
        place_pose_tf = inhand_pose_tf.compose(predicted_place_rel_transform)
        place_pose = place_pose_tf.get_matrix().T.squeeze(-1).detach().cpu().numpy()
        
        # SAVE INFERENCE DATA
        predicted_tf = inhand_pose @ predicted_tf
        print(f"Predicted Pose: \n{np.round(predicted_tf,3)}")
        print(f"Predicted Placement Pose: \n{np.round(place_pose,3)}")
        print(f"InHand Pose: \n{np.round(inhand_pose,3)}")
        np.save(cls.__save_dir + "/predicted_pose.npy", place_pose)
        np.save(cls.__save_dir + "/predicted_tf.npy", predicted_tf)
        np.save(cls.__save_dir + "/inhand_pose.npy", inhand_pose)
        
        pred_place_points = predicted_place_rel_transform.transform_points(action_points)
        np.save(cls.__save_dir + "/pred_place_points.npy", pred_place_points[0].detach().cpu().numpy())
        np.save(cls.__save_dir + "/action_points.npy", action_points[0].detach().cpu().numpy())
        np.save(cls.__save_dir + "/anchor_points.npy", anchor_points[0].detach().cpu().numpy())
        
    
        return place_pose   
    
    @classmethod
    def load_taxpose_model(cls, cfg):
        model_path = cfg.checkpoints.place 
        model_cfg = cfg.place_model 
        wandb_cfg = cfg.wandb
        task_cfg = cfg.place_task
    
        print(f"Loading TaxPose model with config: {model_cfg}")

        ckpt_file = get_weights_path(model_path, wandb_cfg, run=None)

        network = ResidualFlow_DiffEmbTransformer(
            pred_weight=model_cfg.pred_weight,
            emb_nn=model_cfg.emb_nn,
            emb_dims=model_cfg.emb_dims,
            return_flow_component=model_cfg.return_flow_component,
            center_feature=model_cfg.center_feature,
            # inital_sampling_ratio=model_cfg.inital_sampling_ratio,
            residual_on=model_cfg.residual_on,
            multilaterate=model_cfg.multilaterate,
            sample=model_cfg.mlat_sample,
            mlat_nkps=model_cfg.mlat_nkps,
            break_symmetry=model_cfg.break_symmetry,
        )
        model = EquivarianceTrainingModule(
            network,
            weight_normalize=task_cfg.weight_normalize,
            softmax_temperature=task_cfg.softmax_temperature,
            sigmoid_on=True,
            flow_supervision="both",
        )
        weights = torch.load(ckpt_file)["state_dict"]
        model.load_state_dict(weights)

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model

    @classmethod
    def crop_point_cloud(cls, pcd: o3d.geometry.PointCloud, crop_box: Dict[str, float]) -> o3d.geometry.PointCloud:

        points = np.asarray(pcd.points)
        mask = np.logical_and.reduce(
            (points[:, 0] >= crop_box['min_bound'][0],
             points[:, 0] <= crop_box['max_bound'][0],
             points[:, 1] >= crop_box['min_bound'][1],
             points[:, 1] <= crop_box['max_bound'][1],
             points[:, 2] >= crop_box['min_bound'][2],
             points[:, 2] <= crop_box['max_bound'][2])
        )
        points = points[mask]

        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(points)

        return pcd_t

    @classmethod
    def filter_point_cloud(cls, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd
    
    @classmethod
    def __robot_get_eef_pose(cls) -> np.ndarray:
        return cls.__robot.get_eef_pose()

    @classmethod
    def __robot_move_to_pose(cls, pose: np.ndarray) -> None:
        for _ in range(3):
            if not cls.__robot.move_to_pose(
                position=pose[:3, 3], orientation=pose[:3, :3]
            ):
                print(f"Failed to move to pose.")
            else:
                break

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

def prepare_dataset() -> None:
        
    data_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/"
    num_demos = 5      
    
    teach_pcd_dir = os.path.join(data_dir, "teach_data/pcd_data")
    teach_pose_dir = os.path.join(data_dir, "teach_data/pose_data")
    calib_dir = os.path.join(data_dir, "calib_data")
    
    data_template = {
        'action':
            {
                'pcd': None,
                'tf': None,
            },
        'anchor':
            {
                'pcd': None,
                'tf': None,
            }
    }
    
    data_list = [data_template for _ in range(num_demos)]
    
    # Load pose data
    T_base2cam2 = np.load(os.path.join(calib_dir, "T_base2cam2.npy"))
    T_eef2cam3 = np.load(os.path.join(calib_dir, "T_eef2cam3.npy"))        
    for i in range(num_demos):
        target_pose_file = f"demo{i+1}_placement_pose.npy"
        action_pose_file = f"demo{i+1}_gripper_close_up_pose.npy"
        anchor_pose_file = f"demo{i+1}_ih_camera_view_pose.npy" 
        
        target_pose = np.load(os.path.join(teach_pose_dir, target_pose_file))
        action_pose = np.load(os.path.join(teach_pose_dir, action_pose_file))
        anchor_pose = np.load(os.path.join(teach_pose_dir, anchor_pose_file))
        
        T = target_pose @ np.linalg.inv(action_pose) @ T_base2cam2
        T[2,3] = T[2,3] + 0.035
        T[1,3] = T[1,3] + 0.001
        T[0,3] = T[0,3] - 0.003
        data_list[i]['action']['tf'] = T
        data_list[i]['anchor']['tf'] = anchor_pose @ T_eef2cam3        
    
    # Load point cloud data
    for i in range(num_demos):
        anchor_pcd_file = f"demo{i+1}_ih_camera_view_cam3_gripper_pointcloud.ply"
        action_pcd_file= f"demo{i+1}_gripper_close_up_cam2_closeup_pointcloud.ply"
        
        action_pcd = o3d.io.read_point_cloud(os.path.join(teach_pcd_dir, action_pcd_file))
        anchor_pcd = o3d.io.read_point_cloud(os.path.join(teach_pcd_dir, anchor_pcd_file))

        # crop and filter point cloud data
        action_crop_box = {
            'min_bound': np.array([-200, -60, 0.0]),
            'max_bound': np.array([200, 60, 200]),

        }
        action_pcd = ExecutePlace.crop_point_cloud(action_pcd, action_crop_box)
        action_pcd = ExecutePlace.filter_point_cloud(action_pcd)
        action_pcd.points = o3d.utility.Vector3dVector(np.asarray(action_pcd.points)/1000.0)
        
        anchor_crop_box = {
            'min_bound': np.array([-53, -28, 0.0]),
            'max_bound': np.array([53, 25, 300])
        }
        anchor_pcd = ExecutePlace.crop_point_cloud(anchor_pcd, anchor_crop_box)
        anchor_pcd = ExecutePlace.filter_point_cloud(anchor_pcd)
        anchor_pcd.points = o3d.utility.Vector3dVector(np.asarray(anchor_pcd.points)/1000.0)
        
        # transform point cloud data
        anchor_pcd = anchor_pcd.transform(data_list[i]['anchor']['tf'])
        action_pcd = action_pcd.transform(data_list[i]['action']['tf'])
        
        data_list[i]['action']['pcd'] = np.asarray(action_pcd.points)
        data_list[i]['anchor']['pcd'] = np.asarray(anchor_pcd.points)

    action = data_list[0]['action']['pcd']
    anchor = data_list[0]['anchor']['pcd']
    
    return action, anchor

def prepare_dataset2() -> None:
        
    data_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/"
    num_demos = 5
    
    teach_pcd_dir = os.path.join(data_dir, "teach_data/pcd_data")
    teach_pose_dir = os.path.join(data_dir, "teach_data/pose_data")
    calib_dir = os.path.join(data_dir, "calib_data")
    
    data_template = {
        'action':
            {
                'pcd': None,
                'tf': None,
            },
        'anchor':
            {
                'pcd': None,
                'tf': None,
            }
    }
    
    # data_list = [data_template for _ in range(num_demos)]
    data_list = []
    
    # Load pose data
    T_base2cam2 = np.load(os.path.join(calib_dir, "T_base2cam2.npy"))
    T_eef2cam3 = np.load(os.path.join(calib_dir, "T_eef2cam3.npy"))        
        
    for i in range(num_demos):
        data = copy.deepcopy(data_template)
        target_pose_file = f"demo{i+1}_placement_pose.npy"
        action_pose_file = f"demo{i+1}_gripper_close_up_pose.npy"
        anchor_pose_file = f"demo{i+1}_ih_camera_view_pose.npy" 
        
        target_pose = np.load(os.path.join(teach_pose_dir, target_pose_file))
        action_pose = np.load(os.path.join(teach_pose_dir, action_pose_file))
        anchor_pose = np.load(os.path.join(teach_pose_dir, anchor_pose_file))
        
        T = target_pose @ np.linalg.inv(action_pose) @ T_base2cam2
        T[2,3] = T[2,3] + 0.035
        T[1,3] = T[1,3] + 0.001
        T[0,3] = T[0,3] - 0.003
        data['action']['tf'] = T
        data['anchor']['tf'] = anchor_pose @ T_eef2cam3
        
        data_list.append(data)
        
        
    # Load point cloud data
    for i in range(num_demos):
        anchor_pcd_file = f"demo{i+1}_ih_camera_view_cam3_gripper_pointcloud.ply"
        action_pcd_file= f"demo{i+1}_gripper_close_up_cam2_closeup_pointcloud.ply"
        
        action_pcd = o3d.io.read_point_cloud(os.path.join(teach_pcd_dir, action_pcd_file))
        anchor_pcd = o3d.io.read_point_cloud(os.path.join(teach_pcd_dir, anchor_pcd_file))

        # crop and filter point cloud data
        action_crop_box = {
            'min_bound': np.array([-200, -60, 0.0]),
            'max_bound': np.array([200, 60, 200]),

        }
        action_pcd = ExecutePlace.crop_point_cloud(action_pcd, action_crop_box)
        action_pcd = ExecutePlace.filter_point_cloud(action_pcd)
        action_pcd.points = o3d.utility.Vector3dVector(np.asarray(action_pcd.points)/1000.0)
        
        anchor_crop_box = {
            'min_bound': np.array([-53, -28, 0.0]),
            'max_bound': np.array([53, 25, 300])
        }
        anchor_pcd = ExecutePlace.crop_point_cloud(anchor_pcd, anchor_crop_box)
        anchor_pcd = ExecutePlace.filter_point_cloud(anchor_pcd)
        anchor_pcd.points = o3d.utility.Vector3dVector(np.asarray(anchor_pcd.points)/1000.0)
        
        # transform point cloud data
        anchor_pcd = anchor_pcd.transform(data_list[i]['anchor']['tf'])
        action_pcd = action_pcd.transform(data_list[i]['action']['tf'])

        data_list[i]['action']['pcd'] = np.asarray(action_pcd.points)
        data_list[i]['anchor']['pcd'] = np.asarray(anchor_pcd.points)
        
    action = data_list[0]['action']['pcd']
    anchor = data_list[0]['anchor']['pcd']
    
    return action, anchor

def test_taxpose_wp():
    ## Load data
    pcd_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/teach_data"
    gripper_close_up_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, "pcd_data/demo3_gripper_close_up_cam2_closeup_pointcloud.ply"))
    ih_camera_view_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, "pcd_data/demo3_ih_camera_view_cam3_gripper_pointcloud.ply"))
    ih_camera_view_pose = np.load(os.path.join(pcd_dir, "pose_data/demo1_ih_camera_view_pose.npy"))
    
    ExecutePlace.predict_placement_pose_data['gripper_close_up']['cam2_close_up']['pcd'] = gripper_close_up_pcd
    ExecutePlace.predict_placement_pose_data['ih_camera_view']['cam3_gripper']['pcd'] = ih_camera_view_pcd
    ExecutePlace.predict_placement_pose_data['ih_camera_view']['eef_pose'] = ih_camera_view_pose
    
    predicted_pose = ExecutePlace.infer_placement_pose()
    
    breakpoint()

def test2_taxpose_wp():
    batch = torch.load('/home/mfi/repos/rtc_vision_toolbox/test/debug.pt')
    
    print(f'batch keys: {batch.keys()}')
    action_points = batch['points_action_trans'][0]
    anchor_points = batch['points_anchor_trans'][0]
    
    T1 = random_se3(
        1,
        rot_var=np.pi,
        trans_var=2.0,
        device=action_points.device,
        fix_random=False,
        rot_sample_method="quat_uniform",
    )
    
    T2 = random_se3(
        1,
        rot_var=np.pi,
        trans_var=2.0,
        device=action_points.device,
        fix_random=False,
        rot_sample_method="random_upright",
    )
    
    action_points = T1.transform_points(action_points)
    anchor_points = T2.transform_points(anchor_points)
    
    action_points = action_points.detach().cpu().numpy()
    anchor_points = anchor_points.detach().cpu().numpy()
    
    # action_points = np.load('/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/execute_data/0708_1239/action_points.npy')
    # anchor_points = np.load('/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/execute_data/0708_1239/anchor_points.npy')
    
    # action_points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/action.npy")
    # anchor_points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/anchor.npy")
        
    predicted_pose = ExecutePlace.test_infer_placement_pose(action_points, anchor_points)
    
    breakpoint()
    
def test3_taxpose_wp():
    
    # action_points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/debug_action_points_wandb.npy")
    # anchor_points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/debug_anchor_points_wandb.npy")
    
    # action_points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/debug_action_points_demo.npy")
    # anchor_points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/debug_anchor_points_demo.npy")
    
    action_points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/debug_action_points_learn.npy")
    anchor_points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/debug_anchor_points_learn.npy")    
        
    predicted_pose = ExecutePlace.test_infer_placement_pose(action_points, anchor_points)
    
    breakpoint()

def test4_taxpose_wp():
    action, anchor = prepare_dataset()
    
    # np.save("/home/mfi/repos/rtc_vision_toolbox/test/action_0n5_2.npy", action)
    # np.save("/home/mfi/repos/rtc_vision_toolbox/test/anchor_0n5_2.npy", anchor)
    
    # ExecutePlace.test_infer_placement_pose(action, anchor)
    
    breakpoint()

def test_taxpose_dsub():
    ## Load data
    pcd_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-27-dsub/teach_data"
    gripper_close_up_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, "pcd_data/demo4_gripper_close_up_cam2_closeup_pointcloud.ply"))
    ih_camera_view_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, "pcd_data/demo4_ih_camera_view_cam3_gripper_pointcloud.ply"))
    ih_camera_view_pose = np.load(os.path.join(pcd_dir, "pose_data/demo4_ih_camera_view_pose.npy"))
    
    ExecutePlace.predict_placement_pose_data['gripper_close_up']['cam2_close_up']['pcd'] = gripper_close_up_pcd
    ExecutePlace.predict_placement_pose_data['ih_camera_view']['cam3_gripper']['pcd'] = ih_camera_view_pcd
    ExecutePlace.predict_placement_pose_data['ih_camera_view']['eef_pose'] = ih_camera_view_pose
    
    predicted_pose = ExecutePlace.infer_placement_pose()
    
    breakpoint()

@hydra.main(config_path="../models/taxpose/configs", config_name="eval_mfi")
def main(cfg: DictConfig):
    ExecutePlace.setup(cfg, "waterproof")
    # ExecutePlace.setup(cfg, "dsub")
    # ExecutePlace.execute()
    # test2_taxpose_wp()
    # test_taxpose_wp()
    # test3_taxpose_wp()
    test4_taxpose_wp()

if __name__ == "__main__":
    main()