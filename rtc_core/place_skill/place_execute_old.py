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
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from pytorch3d.transforms import Transform3d
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
        self.devices = Devices(cfg.devices)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = os.path.join(current_dir, "../..")

        self.predict_board_pose_data = {
            "image_segmentation_model": None,
            "camera": "cam0_board",
            "view": "out_of_way",
            "rgb": None,
            "depth_data": None,
        }

        self.predict_placement_pose_data = {
            "taxpose_model": self.load_taxpose_model(cfg),
            "action": {
                "camera": cfg.execution.action.camera,
                "view": cfg.execution.action.view,
                "camera_type": cfg.execution.action.camera_type,
                "pcd": None,
                "eef_pose": None,
            },
            "anchor": {
                "camera": cfg.execution.anchor.camera,
                "view": cfg.execution.anchor.view,
                "camera_type": cfg.execution.anchor.camera_type,
                "pcd": None,
                "eef_pose": None,
            }
        }

        self.data_dir: str = os.path.join(self.project_dir, cfg.data_dir)
        self.save_dir: str = None
        self.object: str = None
        self.debug = cfg.debug
        self.object = cfg.object

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

        cam_keys = list(cfg.devices.cameras.keys())
        self.cam_setup = {key: {} for key in cam_keys}

        for key in self.cam_setup.keys():
            if key not in cfg.devices.cameras.keys():
                continue
            for subkey in cfg.devices.cameras[key]['setup'].keys():
                filepath = os.path.join(
                    self.project_dir, cfg.devices.cameras[key]['setup'][subkey])
                self.cam_setup[key][subkey] = np.load(filepath)

        # self.poses = {
        #     "start_pose": np.load(os.path.join(self.data_dir, "poses", "start_pose.npy")),
        #     "out_of_way_pose": np.load(os.path.join(self.data_dir, "poses", "out_of_way_pose.npy")),
        #     "gripper_close_up_pose": np.load(os.path.join(self.data_dir, "poses", "gripper_close_up_pose.npy")),
        #     "pre_placement_offset": np.load(os.path.join(self.data_dir, "learn_data", "pre_placement_offset.npy")),
        # }

        self.poses = {
            "start_pose": np.load(os.path.join(self.data_dir, "poses", "start_pose.npy")),
            "gripper_close_up_pose": np.load(os.path.join(self.data_dir, "poses", "gripper_close_up_pose.npy")),
        }

        print(f"Setup done for {self.object.upper()}")

    def execute(self) -> None:

        print(f"EXECUTING PLACE FOR {self.object.upper()}")

        print("################################################################################")
        print("1. Start Pose")
        print("################################################################################")

        self.devices.robot_move_to_pose(self.poses['start_pose'])
        time.sleep(0.5)

        print("################################################################################")
        print("2. Grasp Object")
        print("################################################################################")

        print("Open gripper")

        self.devices.gripper_open()
        time.sleep(0.5)

        input("Press Enter to close gripper when object inside gripper...")

        self.devices.gripper_close()
        time.sleep(0.5)

        # print("################################################################################")
        # print("3. Out of Way Pose")
        # print("################################################################################")

        # self.devices.robot_move_to_pose(self.poses['out_of_way_pose'])

        # time.sleep(0.5)

        # self.collect_data("out_of_way")
        # time.sleep(0.5)

        print("################################################################################")
        print("4. Gripper Close Up Pose")
        print("################################################################################")

        self.devices.robot_move_to_pose(self.poses['start_pose'])

        print(f"Moving to object in hand close up pose...")
        self.devices.robot_move_to_pose(
            self.poses['gripper_close_up_pose'])

        time.sleep(0.5)

        self.collect_data("gripper_close_up_view")
        time.sleep(0.5)
        breakpoint()
        print("################################################################################")
        print("5. Predict and Move to Approx Pre Placement Pose")
        print("################################################################################")
        # approx_pre_placement_pose = self.infer_board_pose()

        # print(f"Approx Pre Placement Pose: \n{approx_pre_placement_pose}")
        # continue_execution = input(
        #     "Press Enter to continue execution. Press 'c' to debug:")
        # if continue_execution == "c":
        #     breakpoint()

        self.devices.robot_move_to_pose(self.poses['start_pose'])
        # approx_pre_placement_pose = np.load("/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/08-02-wp/teach_data/pose_data/demo0_ih_camera_view0_pose.npy")
        approx_pre_placement_pose = np.load("/home/mfi/repos/rtc_vision_toolbox/notebooks/ih_cam_tf.npy")

        self.devices.robot_move_to_pose(approx_pre_placement_pose)
        self.collect_data("ih_camera_view")

        time.sleep(0.5)
        breakpoint()
        print("################################################################################")
        print("7. Infer Placement Pose and Move to Placement Pose offset")
        print("################################################################################")

        placement_pose = np.dot(self.infer_placement_pose(), np.eye(4))
        
        return
        offset_placement_pose = np.dot(
            placement_pose, self.poses['pre_placement_offset'])
        continue_execution = 'c'
        
        while continue_execution == 'c':
            continue_execution = input(
                "Press Enter to continue execution. Press 'c' to try again with ih_pose:")
            if continue_execution == "c":
                input("Press Enter when in ih_pose...")
                self.collect_data("ih_camera_view")
                placement_pose = np.dot(self.infer_placement_pose(), np.eye(4))
                offset_placement_pose = np.dot(
                    placement_pose, self.poses['pre_placement_offset'])
                        
        self.devices.robot_move_to_pose(offset_placement_pose)
        time.sleep(0.5)

        print("################################################################################")
        print("8. Final Placement")
        print("################################################################################")

        continue_execution = input(
            "Press Enter to continue execution. Press 'c' to debug:")
        if continue_execution == "c":
            breakpoint()

        self.devices.robot_move_to_pose(placement_pose)
        
        input("Press Enter to return to release and start pose continue...")
        time.sleep(0.5)

        print("################################################################################")
        print("9. Return to Start Pose")
        print("################################################################################")

        self.devices.gripper_open()
        time.sleep(0.5)
        self.devices.robot_move_to_pose(self.poses['start_pose'])

        print("################################################################################")
        print(f"\033[1m\033[3m{self.object.upper()} PLACEMENT DONE!\033[0m")

    def infer_board_pose(self) -> np.ndarray:
        self.devices.robot_move_to_pose(self.poses['start_pose'])
        input("Move closer to the placement pose and Press Enter")
        return self.devices.robot_get_eef_pose()

    def infer_placement_pose(self) -> np.ndarray:

        action_pcd_raw = self.predict_placement_pose_data['action']['pcd']
        action_crop_box = self.cfg.execution.action.crop_box

        action_pcd = self.crop_point_cloud(action_pcd_raw, action_crop_box)
        action_pcd = self.filter_point_cloud(action_pcd)
        action_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(action_pcd.points)/1000)

        anchor_pcd_raw = self.predict_placement_pose_data['anchor']['pcd']
        anchor_crop_box = self.cfg.execution.anchor.crop_box
        
        anchor_pcd = self.crop_point_cloud(anchor_pcd_raw, anchor_crop_box)
        anchor_pcd = self.filter_point_cloud(anchor_pcd)
        anchor_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(anchor_pcd.points)/1000)

        def get_transform(type: str) -> np.ndarray:
            if self.cfg.execution[type].camera_type == "on_base":
                return self.cam_setup[self.cfg.execution[type].camera][f"T_base2cam"]
            elif self.cfg.execution[type].camera_type == "on_hand":
                T_base2eef = self.predict_placement_pose_data[type]['eef_pose']
                T_eef2cam = self.cam_setup[self.cfg.execution[type].camera][f"T_eef2cam"]
                return np.dot(T_base2eef, T_eef2cam)
        
        action_pcd_tf = action_pcd.transform(get_transform('action'))
        anchor_pcd_tf = anchor_pcd.transform(get_transform('anchor'))
        
        np.save(self.save_dir + "/anchor_final.npy", np.asarray(anchor_pcd_tf.points))

        action_points = np.asarray(action_pcd_tf.points).astype(np.float32)
        anchor_points = np.asarray(anchor_pcd_tf.points).astype(np.float32)
        place_pose = self.infer_taxpose(action_points, anchor_points)

        place_pose = np.dot(place_pose, np.eye(4))
        place_pose = place_pose + self.cfg.execution.target.bias

        return place_pose

    def infer_placement_pose_old(self) -> np.ndarray:

        action_pcd_raw = self.predict_placement_pose_data['action']['pcd']
        action_crop_box = {
            'min_bound': np.array([-200, -60, 0.0]),
            'max_bound': np.array([200, 60, 200]),
        }
        action_pcd = self.crop_point_cloud(action_pcd_raw, action_crop_box)
        action_pcd = self.filter_point_cloud(action_pcd)
        action_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(action_pcd.points)/1000)

        anchor_pcd_raw = self.predict_placement_pose_data['anchor']['pcd']
        anchor_crop_box = {
            'min_bound': np.array([-53, -28, 0.0]),
            'max_bound': np.array([53, 25, 300])
        }
        
        anchor_pcd = self.crop_point_cloud(anchor_pcd_raw, anchor_crop_box)
        anchor_pcd = self.filter_point_cloud(anchor_pcd)
        anchor_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(anchor_pcd.points)/1000)

        T_base2eef = self.predict_placement_pose_data['anchor']['eef_pose']
        T_base2cam_anchor = np.dot(
            T_base2eef, self.cam_setup['cam3_gripper']['T_eef2cam'])
        T_base2cam_action = self.cam_setup['cam2_closeup']['T_base2cam']
        action_pcd_tf = action_pcd.transform(T_base2cam_action)
        anchor_pcd_tf = anchor_pcd.transform(T_base2cam_anchor)
        
        np.save(self.save_dir + "/anchor_final.npy", np.asarray(anchor_pcd_tf.points))

        action_points = np.asarray(action_pcd_tf.points).astype(np.float32)
        anchor_points = np.asarray(anchor_pcd_tf.points).astype(np.float32)
        place_pose = self.infer_taxpose(action_points, anchor_points)

        return place_pose

    def infer_taxpose(self, action_points, anchor_points):
        action_points = maybe_downsample(action_points[None, ...], 2048, 'fps')
        action_points = torch.from_numpy(action_points).to('cuda')

        anchor_points = maybe_downsample(anchor_points[None, ...], 2048, 'fps')
        anchor_points = torch.from_numpy(anchor_points).cuda('cuda')

        print(f'action_points: {action_points[:, :3]}')
        print(f'anchor_points: {anchor_points[:, :3]}')

        place_model = self.predict_placement_pose_data['taxpose_model']

        place_out = place_model(action_points, anchor_points, None, None)
        predicted_place_rel_transform = place_out['pred_T_action']

        inhand_pose = self.poses['gripper_close_up_pose']
        inhand_pose_tf = Transform3d(
            matrix=torch.Tensor(inhand_pose.T),
            device=predicted_place_rel_transform.device
        ).to(predicted_place_rel_transform.device)
        place_pose_tf = inhand_pose_tf.compose(predicted_place_rel_transform)
        place_pose = place_pose_tf.get_matrix().T.squeeze(-1).detach().cpu().numpy()

        # SAVE INFERENCE DATA
        # predicted_tf = inhand_pose @ predicted_tf
        print(f"Predicted Placement Pose: \n{np.round(place_pose,3)}")
        np.save(self.save_dir + "/predicted_pose.npy", place_pose)

        pred_place_points = predicted_place_rel_transform.transform_points(
            action_points)
        np.save(self.save_dir + "/pred_place_points.npy",
                pred_place_points[0].detach().cpu().numpy())
        np.save(self.save_dir + "/action_points.npy",
                action_points[0].detach().cpu().numpy())
        np.save(self.save_dir + "/anchor_points.npy",
                anchor_points[0].detach().cpu().numpy())

        return place_pose

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

        Args:
            robot_state (str): The current state of the robot.

        Returns:
            None
        """

        # char = input(f"Press Enter to collect data for {robot_state}...")
        # if char == "n":
        #     return
        
        max_depth = 1500
        if robot_state in ["gripper_close_up_view", "ih_camera_view"]:
            max_depth = 305  # 254 mm = 10 inches, 305 mm = 12 inches

        data_dir = self.save_dir

        # Collect data from cameras
        for cam_name in self.cam_setup.keys():
            print(f"Collecting data from {cam_name} with max depth {max_depth}...")
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
            target_dict_list = [self.predict_board_pose_data,
                                self.predict_placement_pose_data["action"],
                                self.predict_placement_pose_data["anchor"]]

            for target_dict in target_dict_list:
                if target_dict["camera"] == cam_name and target_dict["view"] == robot_state:
                    for key in target_dict.keys():
                        if key == "pcd":
                            target_dict[key] = point_cloud
                        if key == "rgb":
                            target_dict[key] = image
                        if key == "depth_data":
                            target_dict[key] = depth_data

        # Collect data from robot
        eef_pose = self.devices.robot_get_eef_pose()
        poses_folder = os.path.join(data_dir, "pose_data")
        if not os.path.exists(poses_folder):
            os.makedirs(poses_folder)
        np.save(os.path.join(poses_folder,
                f"{robot_state}_pose.npy"), eef_pose)

        # get data for prediction
        target_dict_list = [self.predict_board_pose_data,
                            self.predict_placement_pose_data["action"],
                            self.predict_placement_pose_data["anchor"]]

        for target_dict in target_dict_list:
            if target_dict["camera"] == cam_name and target_dict["view"] == robot_state:
                for key in target_dict.keys():
                    if key == "eef_pose":
                        target_dict[key] = eef_pose

    def crop_point_cloud(self, pcd: o3d.geometry.PointCloud, crop_box: Dict[str, float]) -> o3d.geometry.PointCloud:

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

    def filter_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd
