import argparse
import os
import sys

import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

from rtc_core.place_skill.place_execute import ExecutePlace


def test_taxpose_wp(place_execute: ExecutePlace):
    ## Load data
    # pcd_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/06-20-wp/teach_data"
    # gripper_close_up_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, "pcd_data/demo3_gripper_close_up_cam2_closeup_pointcloud.ply"))
    # ih_camera_view_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, "pcd_data/demo3_ih_camera_view_cam3_gripper_pointcloud.ply"))
    # ih_camera_view_pose = np.load(os.path.join(pcd_dir, "pose_data/demo1_ih_camera_view_pose.npy"))

    pcd_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/08-14-wp/execute_data/0826_1703"
    gripper_close_up_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, "pcd_data/gripper_close_up_view_cam2_closeup_pointcloud.ply"))
    gripper_close_up_pose = np.load(os.path.join(pcd_dir, "pose_data/gripper_close_up_view_pose.npy"))

    pcd_dir = "/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/08-14-wp/execute_data/0826_1703"    
    ih_camera_view_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, "pcd_data/ih_camera_view_cam3_gripper_pointcloud.ply"))
    ih_camera_view_pose = np.load(os.path.join(pcd_dir, "pose_data/ih_camera_view_pose.npy"))
       
    place_execute.predict_placement_pose_data['action']['pcd'] = gripper_close_up_pcd
    place_execute.predict_placement_pose_data['anchor']['pcd'] = ih_camera_view_pcd
    place_execute.predict_placement_pose_data['anchor']['eef_pose'] = ih_camera_view_pose
    place_execute.predict_placement_pose_data['action']['eef_pose'] = gripper_close_up_pose
    
    predicted_pose = place_execute.infer_placement_pose()    

def test_taxpose(place_execute: ExecutePlace):
    
    time = '0826_1703'
    action = np.load(f'/home/mfi/repos/rtc_vision_toolbox/test/points.npy')
    anchor = np.load(f'/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/08-14-wp/execute_data/{time}/anchor_points.npy')
    
    # # get random se3 pose
    # rotation = R.from_euler('xyz', np.random.uniform(-np.pi, np.pi, 3)).as_matrix()
    # translation = np.random.uniform(-0.1, 0.1, 3)
    # se3_pose = np.eye(4)
    # se3_pose[:3, :3] = rotation
    # se3_pose[:3, 3] = translation
    
    # action = np.hstack([action, np.ones((action.shape[0], 1))])
    # action = np.dot(action, se3_pose.T)
    # action = action[:, :3]
    
    action[:, 2] = -action[:, 2]
    anchor[:, 2] = -anchor[:, 2]
    place_execute.infer_taxpose(action, anchor)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args, unknown = parser.parse_known_args()
    config_file = args.config

    # Filter out custom arguments before Hydra processes them
    filtered_argv = [arg for arg in sys.argv if arg.startswith("--hydra")]

    # Manually call Hydra main
    sys.argv = [sys.argv[0]] + filtered_argv        
    
    # Read configuration file

    config_dir = os.path.dirname(config_file)
    config_name = os.path.basename(config_file)
    
    print(f"Reading configuration file: {config_file}")
    
    hydra.initialize(config_path=config_dir, version_base="1.3")
    config: DictConfig = hydra.compose(config_name)
    
    # print(OmegaConf.to_yaml(config, resolve=True))

    place_execute = ExecutePlace(config)

    place_execute.validate_execute_repeat()
    # place_execute.execute()    