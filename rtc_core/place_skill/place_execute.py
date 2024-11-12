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
from hydra.core.global_hydra import GlobalHydra
from inputimeout import TimeoutOccurred, inputimeout
from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import Transform3d
from scipy.spatial.transform import Rotation as R
from taxpose.datasets.augmentations import maybe_downsample
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import \
    EquivarianceTrainingModule
from taxpose.utils.load_model import get_weights_path
from taxpose.utils.se3 import random_se3

from rtc_core.devices.devices import Devices

# CONFIG_DIR = "/home/mfi/repos/rtc_vision_toolbox/config/"
# DATA_DIR = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/"

# INHAND_DEVICES_SN = ["CL8FC3100NM"]
# BOARD_DEVICES_SN = ["CL8FC3100RL", "CL8FC3100W3"]
# ROBOT_IP = "172.26.179.142"
# ROBOT_NAME = "yk_builder"
# GRIPPER_PORT = "/dev/ttyUSB0"

# T_eef2camera_filepath = "/home/mfi/repos/rtc_vision_toolbox/data/calibration_data/rs/T_eef2camera.npy"
# T_base2cam2_filepath = "/home/mfi/repos/rtc_vision_toolbox/data/calibration_data/zed/T_base2cam2.npy"


class ExecutePlace:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.debug = cfg.debug
        self.devices = Devices(cfg.devices, cfg.debug)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = os.path.join(current_dir, "../..")

        self.predict_placement_pose_data = {
            "taxpose_model": self.load_taxpose_model(cfg),
            "action": {
                "camera": cfg.execution.action.camera,
                "view": cfg.execution.action.view,                
                "camera_type": cfg.devices.cameras[cfg.execution.action.camera]['type'],
                "pcd": None,
                "eef_pose": None,
            },
            "anchor": {
                "camera": cfg.execution.anchor.camera,
                "view": cfg.execution.anchor.view,
                "camera_type": cfg.devices.cameras[cfg.execution.anchor.camera]['type'],
                "pcd": None,
                "eef_pose": None,
            }
        }

        self.data_dir: str = os.path.join(self.project_dir, cfg.data_dir)
        self.save_dir: str = None
        self.object: str = cfg.object

        now = datetime.datetime.now().strftime("%m%d_%H%M")
        self.save_dir = os.path.join(self.data_dir, f"execute_data/{now}")

        if not os.path.exists(self.data_dir):
            raise Exception(
                f"Data directory {self.data_dir} does not exist.")
        else:
            print(f"Data directory: {self.data_dir}")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Save directory: {self.save_dir}")

        self.cam_setup = {key: {} for key in list(cfg.devices.cameras.keys())}

        for key in self.cam_setup.keys():
            for subkey in cfg.devices.cameras[key]['setup'].keys():
                filepath = os.path.join(
                    self.project_dir, cfg.devices.cameras[key]['setup'][subkey])
                self.cam_setup[key][subkey] = np.load(filepath)

        self.poses = {
            "home_pose": np.load(os.path.join(self.data_dir, "poses", "home_pose.npy")),
            "ih_camera_view_pose": np.load(os.path.join(self.data_dir, "poses", "ih_camera_view_pose.npy")),
            "target_pose": np.load(os.path.join(self.data_dir, "poses", "target_pose.npy")),
        }

        print(f"Setup done for {self.object.upper()}")

    def execute(self) -> None:

        print(f"EXECUTING PLACE FOR {self.object.upper()}")

        print("####################################################################")
        print("1. HOME POSE")
        print("####################################################################")

        self.devices.robot_move_to_pose(self.poses['home_pose'], 1, 1)
        
        print("####################################################################")
        print("2. GRASP OBJECT")
        print("####################################################################")

        print("Open gripper")

        self.devices.gripper_open()
        time.sleep(0.5)

        input("Press Enter to close gripper when object inside gripper...")

        self.devices.gripper_close()
        time.sleep(0.5)

        retry = True
        retry_ctr = 0
        
        while retry:
            print("####################################################################")
            print("3. GRIPPER CLOSE UP VIEW")
            print("####################################################################")
            
            print(f"Moving to object in hand close up pose...")
            T_base2camera = self.cam_setup[self.cfg.training.action.camera]["T_base2cam"]
            distance = self.cfg.execution.action.viewing_distance
            T_camera2gripper = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, distance],
                [0,  0,  0, 1]])
            T_eef2gripper = np.asarray(self.devices.gripper.T_ee2gripper)
            T_base2gripper = (T_base2camera @ T_camera2gripper) @ np.linalg.inv(T_eef2gripper)
            gripper_close_up_pose = T_base2gripper

            self.devices.robot_move_to_pose(gripper_close_up_pose, 1, 1)
            self.collect_data("gripper_close_up_view")
                    
            print("####################################################################")
            print("3. IN-HAND CAMERA VIEW")
            print("####################################################################")
            
            self.devices.robot_move_to_pose(self.poses['home_pose'])
            
            print(f"Moving to in-hand camera view pose...")
            self.devices.robot_move_to_pose(self.poses['ih_camera_view_pose'], 1, 1)
            
            self.collect_data("ih_camera_view")

            print("####################################################################")
            print("5. PREDICTING PLACEMENT POSE")
            print("####################################################################")
            
            placement_pose = self.infer_placement_pose()
            pre_placement_pose = placement_pose.copy()
            pre_placement_pose[2, 3] = placement_pose[2, 3] + self.cfg.execution.target.pull_distance
            
            np.save(self.save_dir + "/pose_data/placement_pose.npy", placement_pose)
            
            print("####################################################################")
            print("6. PERFORM PLACEMENT")
            print("####################################################################")
            
            print("Moving to pre-target pose...")
            
            pre_target_pose = self.poses['target_pose'].copy()
            pre_target_pose[2, 3] = self.poses['home_pose'][2, 3]
            self.devices.robot_move_to_pose(pre_target_pose, 1, 1)
            
            input("Press Enter to continue...")
            
            self.devices.robot_move_to_pose(pre_placement_pose, 1, 1)
        
            retry_input = input("Press 'r' to retry or Enter to continue...")
            retry = retry_input == 'r'
            if retry:
                retry_ctr += 1
                self.save_dir = self.save_dir + "_r" + str(retry_ctr)
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                    print(f"Save directory: {self.save_dir}")
            
        self.devices.robot_move_to_pose(placement_pose, 0.05, 0.05)
        
        print("####################################################################")
        print("7. FINISHING UP")
        print("####################################################################")
        
        self.devices.gripper_open()
        time.sleep(0.5)
        self.devices.robot_move_to_pose(self.poses['home_pose'], 1, 1)

        print(f"\033[1m\033[3m{self.object.upper()} PLACEMENT DONE!\033[0m")

    def infer_placement_pose(self) -> np.ndarray:
        
        T_ee2target = np.asarray(self.devices.gripper.T_ee2gripper)

        # PREPARE DATA FOR INFERENCE: ACTION POINTCLOUD
        action_pcd = self.predict_placement_pose_data['action']['pcd']
        action_points = np.asarray(action_pcd.points) / 1000
        action_pcd.points = o3d.utility.Vector3dVector(action_points)
        action_pcd = self.filter_point_cloud(action_pcd)
        
        T_base2action = self.predict_placement_pose_data['action']['eef_pose']
        T_base2cam = self.cam_setup[self.cfg.execution.action.camera][f"T_base2cam"]
        action_tf = np.linalg.inv(T_ee2target) @ (np.linalg.inv(T_base2action) @ T_base2cam)
        action_pcd = action_pcd.transform(action_tf)
        
        action_points = np.asarray(action_pcd.points)
        action_points = action_points[action_points[:, 2] > 0]
        action_points = action_points[action_points[:, 0] < 0.1]
        action_points = action_points[action_points[:, 0] > -0.1]
        
        # PREPARE DATA FOR INFERENCE: ANCHOR POINTCLOUD
        anchor_pcd = self.predict_placement_pose_data['anchor']['pcd']
        anchor_points = np.asarray(anchor_pcd.points) / 1000
        anchor_pcd.points = o3d.utility.Vector3dVector(anchor_points)
              
        T_eef2cam = self.cam_setup[self.cfg.execution.anchor.camera][f"T_eef2cam"]
        T_base2anchor = self.predict_placement_pose_data['anchor']['eef_pose']
        T_base2cam = T_base2anchor @ T_eef2cam
        T_base2target = self.poses['target_pose'] @ T_ee2target
        
        crop_tf = np.linalg.inv(T_base2target) @ T_base2cam
        anchor_pcd = anchor_pcd.transform(crop_tf)
        anchor_bounds = self.cfg.execution.anchor.object_bounds
        object_bounds = {
            'min': np.array(anchor_bounds.min)/1000,
            'max': np.array(anchor_bounds.max)/1000
        }
        anchor_pcd = self.crop_point_cloud(anchor_pcd, object_bounds)
        anchor_points = np.asarray(anchor_pcd.points)
        np.save(self.save_dir + "/anchortest.npy", np.asarray(anchor_pcd.points))
              
        # INFER PLACEMENT POSE
        taxpose_output = self.infer_taxpose(action_points, anchor_points)
        taxpose_output = np.dot(taxpose_output, np.eye(4))
        
        place_pose = T_base2target @ taxpose_output @ np.linalg.inv(T_ee2target)
        
        return place_pose

    def infer_taxpose(self, action_points, anchor_points):
        
        # flip z axis for taxpose
        action_points[:,2] = -action_points[:,2]
        anchor_points[:,2] = -anchor_points[:,2]

        if self.debug:
            print(f"Action points: {action_points.shape}")
            print(f"Anchor points: {anchor_points.shape}")
                  
        action_points = maybe_downsample(action_points[None, ...], 2048, 'fps')
        action_points = torch.from_numpy(action_points).to('cuda')

        anchor_points = maybe_downsample(anchor_points[None, ...], 2048, 'fps')
        anchor_points = torch.from_numpy(anchor_points).cuda('cuda')

        if self.debug:
            print(f'action_points: {action_points[:, :3]}')
            print(f'anchor_points: {anchor_points[:, :3]}')

        place_model = self.predict_placement_pose_data['taxpose_model']

        place_out = place_model(action_points, anchor_points, None, None)
        predicted_place_rel_transform = place_out['pred_T_action']

        # inhand_pose = self.poses['gripper_close_up_pose']
        # inhand_pose_tf = Transform3d(
        #     matrix=torch.Tensor(inhand_pose.T),
        #     device=predicted_place_rel_transform.device
        # ).to(predicted_place_rel_transform.device)
        # place_pose_tf = inhand_pose_tf.compose(predicted_place_rel_transform)
        # place_pose = place_pose_tf.get_matrix().T.squeeze(-1).detach().cpu().numpy()

        # SAVE INFERENCE DATA
        # predicted_tf = inhand_pose @ predicted_tf
        # print(f"Predicted Placement Pose: \n{np.round(place_pose,3)}")
        # np.save(self.save_dir + "/predicted_pose.npy", place_pose)

        predicted_tf = predicted_place_rel_transform.get_matrix().T.squeeze(-1).detach().cpu().numpy()
        
        pred_place_points = predicted_place_rel_transform.transform_points(
            action_points)
        np.save(self.save_dir + "/pred_place_points.npy",
                pred_place_points[0].detach().cpu().numpy())
        np.save(self.save_dir + "/action_points.npy",
                action_points[0].detach().cpu().numpy())
        np.save(self.save_dir + "/anchor_points.npy",
                anchor_points[0].detach().cpu().numpy())
               
        # fix for flipping z
        T = np.eye(4)
        T[2,2] = -1
        predicted_tf = T @ predicted_tf @ T

        np.save(self.save_dir + "/predicted_tf.npy", predicted_tf)

        return predicted_tf

    def load_taxpose_model(self, config: DictConfig):

        # load taxpose config

        cfg: DictConfig = None
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(
            self.project_dir, config.execution.model_config.config_path)
        relative_config_path = os.path.relpath(config_path, script_dir)
        config_name = config.execution.model_config.config_name

        GlobalHydra.instance().clear()

        @hydra.main(config_path=relative_config_path, config_name=config_name, version_base=None)
        def config(config: DictConfig):
            nonlocal cfg
            resolved_conf = OmegaConf.to_container(config, resolve=True)
            resolved_conf = OmegaConf.create(resolved_conf)
            cfg = resolved_conf
        config()

        # load taxpose model
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

    def collect_data(self, robot_state: str) -> None:
        """
        Collects data from cameras and the robot for a given robot state.
        """
        
        max_depth = 1500
        if robot_state in ["gripper_close_up_view", "ih_camera_view"]:
            max_depth = 305  # 254 mm = 10 inches, 305 mm = 12 inches

        data_dir = self.save_dir

        # Collect data from cameras
        for cam_name in self.cam_setup.keys():
            if (cam_name == "cam2_closeup" and robot_state == "ih_camera_view") or \
               (cam_name == "cam3_gripper" and robot_state == "gripper_close_up_view"):
                   continue
            print(f"Collecting data from {cam_name} with max depth {max_depth}...")
            # image = self.devices.cam_get_rgb_image(cam_name)
            # depth_data = self.devices.cam_get_raw_depth_data(
            #     cam_name, max_depth=max_depth)
            # depth_image = self.devices.cam_get_colormap_depth_image(
            #     cam_name, max_depth=max_depth)
            point_cloud = self.devices.cam_get_point_cloud(
                cam_name, max_mm=max_depth)

            # save images in img folder
            img_folder = os.path.join(data_dir, "img_data")
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            # cv2.imwrite(
            #     os.path.join(
            #         img_folder, f"{robot_state}_{cam_name}_rgb.png"), image
            # )
            # cv2.imwrite(
            #     os.path.join(
            #         img_folder, f"{robot_state}_{cam_name}_depth.png"),
            #     depth_image,
            # )
            # np.save(
            #     os.path.join(
            #         img_folder, f"{robot_state}_{cam_name}_depth_data.npy"),
            #     depth_data,
            # )

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
            target_dict_list = [self.predict_placement_pose_data["action"],
                                self.predict_placement_pose_data["anchor"]]

            for target_dict in target_dict_list:
                if target_dict["camera"] == cam_name and target_dict["view"] == robot_state:
                    for key in target_dict.keys():
                        if key == "pcd":
                            target_dict[key] = point_cloud
                        # if key == "rgb":
                        #     target_dict[key] = image
                        # if key == "depth_data":
                        #     target_dict[key] = depth_data

        # Collect data from robot
        eef_pose = self.devices.robot_get_eef_pose()
        poses_folder = os.path.join(data_dir, "pose_data")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)
        np.save(os.path.join(poses_folder,
                f"{robot_state}_pose.npy"), eef_pose)

        # get data for prediction
        target_dict_list = [self.predict_placement_pose_data["action"],
                            self.predict_placement_pose_data["anchor"]]

        for target_dict in target_dict_list:
            if target_dict["view"] == robot_state:
                for key in target_dict.keys():
                    if key == "eef_pose":
                        target_dict[key] = eef_pose
                        print(f"Setting {key} for {cam_name} and {robot_state}...")

    def crop_points(self, points, crop_box) -> np.ndarray:

        mask = np.logical_and.reduce(
            (points[:, 0] >= crop_box['min'][0],
                points[:, 0] <= crop_box['max'][0],
                points[:, 1] >= crop_box['min'][1],
                points[:, 1] <= crop_box['max'][1],
                points[:, 2] >= crop_box['min'][2],
                points[:, 2] <= crop_box['max'][2])
        )
        points = points[mask]

        return points

    def crop_point_cloud(self, pcd: o3d.geometry.PointCloud, crop_box: Dict[str, float]) -> o3d.geometry.PointCloud:

        points = np.asarray(pcd.points)
        mask = np.logical_and.reduce(
            (points[:, 0] >= crop_box['min'][0],
             points[:, 0] <= crop_box['max'][0],
             points[:, 1] >= crop_box['min'][1],
             points[:, 1] <= crop_box['max'][1],
             points[:, 2] >= crop_box['min'][2],
             points[:, 2] <= crop_box['max'][2])
        )
        points = points[mask]

        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(points)

        return pcd_t

    def filter_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        return pcd

    def validate_execute(self, eef_pose = None) -> None:

        print(f"EXECUTING PLACE FOR {self.object.upper()}")

        print("####################################################################")
        print("1. HOME POSE")
        print("####################################################################")

        self.devices.robot_move_to_pose(self.poses['home_pose'])
        
        print("####################################################################")
        print("2. GRASP OBJECT")
        print("####################################################################")

        print("Open gripper")

        self.devices.gripper_open()
        time.sleep(0.5)
        
        self.devices.robot_move_to_pose(self.poses['target_pose'], 1, 1)

        if eef_pose is None:
            input("Press Enter to close gripper when object inside gripper...")
        else:
            self.devices.robot_move_to_pose(eef_pose, 1, 1)
            
        self.devices.gripper_close()
        time.sleep(0.5)
        
        ground_truth = self.devices.robot_get_eef_pose()
        time.sleep(0.5)

        pre_placement_gt = np.copy(ground_truth)
        pre_placement_gt[2, 3] = pre_placement_gt[2,3] + self.cfg.execution.target.pull_distance
        self.devices.robot_move_to_pose(pre_placement_gt, 1, 1)
        
        retry = True
        retry_ctr = 0
        
        while retry:
            self.devices.robot_move_to_pose(self.poses['home_pose'], 1, 1)

            print("####################################################################")
            print("3. GRIPPER CLOSE UP VIEW")
            print("####################################################################")
            
            print(f"Moving to object in hand close up pose...")
            T_base2camera = self.cam_setup[self.cfg.training.action.camera]["T_base2cam"]

            distance = self.cfg.execution.action.viewing_distance
            T_camera2gripper = np.asarray(self.cfg.devices.gripper.T_camera2gripper)
            T_eef2gripper = np.asarray(self.devices.gripper.T_ee2gripper)
            T_base2gripper = (T_base2camera @ T_camera2gripper) @ np.linalg.inv(T_eef2gripper)
            gripper_close_up_pose = T_base2gripper

            self.devices.robot_move_to_pose(gripper_close_up_pose, 1, 1)
            self.collect_data("gripper_close_up_view")
                    
            print("####################################################################")
            print("3. IN-HAND CAMERA VIEW")
            print("####################################################################")
            
            self.devices.robot_move_to_pose(self.poses['home_pose'], 1, 1)
            
            print(f"Moving to in-hand camera view pose...")
            self.devices.robot_move_to_pose(self.poses['ih_camera_view_pose'], 1, 1)
            
            self.collect_data("ih_camera_view")
            # breakpoint()
            print("####################################################################")
            print("5. PREDICTING PLACEMENT POSE")
            print("####################################################################")
            
            placement_pose = self.infer_placement_pose()
            pre_placement_pose = placement_pose.copy()
            pre_placement_pose[2, 3] = placement_pose[2, 3] + self.cfg.execution.target.pull_distance
            
            # save ground truth and predicted placement pose
            np.save(self.save_dir + "/pose_data/ground_truth.npy", ground_truth)
            np.save(self.save_dir + "/pose_data/placement_pose.npy", placement_pose)
            
            print("####################################################################")
            print("6. PERFORM PLACEMENT")
            print("####################################################################")
            
            print("Moving to pre-target pose...")
            
            pre_target_pose = self.poses['target_pose'].copy()
            pre_target_pose[2, 3] = self.poses['home_pose'][2, 3]
            self.devices.robot_move_to_pose(pre_target_pose, 1, 1)
            
            # input("Press Enter to continue...")
            
            self.devices.robot_move_to_pose(pre_placement_pose)
                        
            rot_error = ((ground_truth) @ np.linalg.inv(placement_pose))[:3,:3]
            euler = R.from_matrix(rot_error).as_euler('xyz', degrees=True)
            rot_error = np.round(np.max(np.abs(euler)),2)
            t_error = np.linalg.norm(ground_truth[:3,3] - placement_pose[:3,3])*1000
            t_error2 = np.linalg.norm(ground_truth[:2,3] - placement_pose[:2,3])*1000
            print(f"\nRotation error: {rot_error}\u00B0,\tTranslation error: {np.round(t_error,2)}, {np.round(t_error2,2)} mm\n")
            
            retry = False
            # retry_input = input("Press 'r' to retry or Enter to continue...")
            # retry = retry_input == 'r'
            # if retry:
            #     retry_ctr += 1
            #     self.save_dir = self.save_dir + "_r" + str(retry_ctr)
            #     if not os.path.exists(self.save_dir):
            #         os.makedirs(self.save_dir)
            #         print(f"Save directory: {self.save_dir}")
                  
        self.devices.robot_move_to_pose(placement_pose, 0.05, 0.05)
        
        print("####################################################################")
        print("7. FINISHING UP")
        print("####################################################################")
        
        self.devices.gripper_open()
        time.sleep(0.5)
        self.devices.robot_move_to_pose(self.poses['home_pose'])

        # create and write results to success.txt file
        success = input("Was the placement successful? (y/n) ")
        with open(os.path.join(self.save_dir, "success.log"), "w") as f:
            f.write(f"{success}\n")    
        if success == 'n':
            input("Reset and preses Enter to continue...")
        
        print(f"\033[1m\033[3m{self.object.upper()} PLACEMENT DONE!\033[0m")

    def validate_execute_repeat(self) -> None:
        
        count = int(input("Enter number of repetitions: "))  
        Ry_range = float(input("Enter rotational variance (+-deg): "))
        x_range = float(input("Enter x variance (+-mm): "))
        z_range = float(input("Enter z variance (+-mm): "))    
        
        T_ee2target = np.asarray(self.cfg.devices.gripper.T_ee2gripper)          
        
        for i in range(count):
            print("####################################################################")
            print(f"REPEAT {i+1}")
            print("####################################################################")
                       
            # add random Ry (+- 10deg), x(+- 5mm), z(+- 10mm)
            Ry = np.random.uniform(-Ry_range, Ry_range)
            x = np.random.uniform(-x_range, x_range)/1000
            z = np.random.uniform(0, z_range)/1000
            
            gripper_pose = np.dot(self.poses['target_pose'], T_ee2target)
            rot = R.from_matrix(gripper_pose[:3,:3]).as_euler('xyz', degrees=True)
            rot[1] = rot[1] + Ry
            gripper_pose[:3,:3] = R.from_euler('xyz', rot, degrees=True).as_matrix()          
            eef_pose = gripper_pose @ np.linalg.inv(T_ee2target)
            eef_pose[2, 3] = eef_pose[2, 3] + z
            eef_pose[0, 3] = eef_pose[0, 3] + x
            
            print(f"Executing with random Ry: {np.round(Ry,2)}\u00B0, x: {np.round(x*1000,2)} mm, z: {np.round(z*1000,2)} mm")
            
            self.validate_execute(eef_pose = eef_pose)
            
            now = datetime.datetime.now().strftime("%m%d_%H%M")
            self.save_dir = os.path.join(self.data_dir, f"execute_data/{now}")

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                print(f"Save directory: {self.save_dir}")
