import argparse
import numpy as np
import open3d as o3d
import os
import datetime
from typing import List, Tuple, Dict

from camera.orbbec.ob_camera import OBCamera

INHAND_DEVICES_SN = ['CL8FC3100NM']
BOARD_DEVICES_SN = ['CL8FC3100RL', 'CL8FC3100W3']

import logging

# Set the logging level to a higher level than INFO (e.g., WARNING or ERROR)
logging.basicConfig(level=logging.WARNING)


def setup_cameras() -> Tuple[Dict[str, OBCamera], Dict[str, OBCamera]]:
    in_hand_cameras = {}
    for sn in INHAND_DEVICES_SN:
        in_hand_cameras[sn] = OBCamera(serial_no=sn)
    
    board_cameras = {}
    for sn in BOARD_DEVICES_SN:
        board_cameras[sn] = OBCamera(serial_no=sn)

    return in_hand_cameras, board_cameras

def collect_camera_point_clouds(cameras: Dict[str, OBCamera]) -> Dict[str, o3d.geometry.PointCloud]:
    point_clouds = {}
    for key, camera in cameras.items():
        point_cloud = camera.get_point_cloud(
            min_mm = 0,
            max_mm = 5000,
            save_points = False,
            use_new_frame = True
        )
        point_clouds[key] = point_cloud
    return point_clouds

def collect_demonstration(args: argparse.Namespace) -> None:
    in_hand_cameras, board_cameras = setup_cameras()

    # Collect data from cameras (point clouds)
    big_object_pcd_dict = collect_camera_point_clouds({**board_cameras, **in_hand_cameras})

    # Save point clouds as .npz files
    demo_path = os.path.join(args.pcd_save_path)#, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(demo_path, exist_ok=True)
    for key, pcd in big_object_pcd_dict.items():
        np.savez(os.path.join(demo_path, f"{key}_big_object.npz"), np.array(pcd.points))

def get_args():
    parser = argparse.ArgumentParser(description="Collect PCDs")
    parser.add_argument("--pcd_save_path", type=str, help="Path to save demonstration data")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    collect_demonstration(args)
    