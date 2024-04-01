import argparse
import numpy as np
import open3d as o3d
from typing import List, Tuple

from camera.orbbec.ob_camera import OBCamera
from robot.ros_robot.ros_robot import ROSRobot

INHAND_DEVICE_IDS = [2]
# BOARD_DEVICE_IDS = [0, 1]
BOARD_DEVICE_IDS = [1]

ROBOT_IP = "172.26.179.142"
ROBOT_NAME = "yk_builder"

def setup_cameras() -> Tuple[List[OBCamera], List[OBCamera]]:
    in_hand_cameras = [OBCamera(device_index=index) for index in INHAND_DEVICE_IDS]
    board_cameras = [OBCamera(device_index=index) for index in BOARD_DEVICE_IDS]

    return in_hand_cameras, board_cameras

def setup_robot() -> ROSRobot:
    robot = ROSRobot(
        rosmaster_ip=ROBOT_IP, 
        robot_name=ROBOT_NAME
    )
    return robot

def merge_point_clouds(
        point_clouds: List[o3d.geometry.PointCloud], 
        camera_to_shared_frame_transforms: List[np.ndarray]
    ) -> o3d.geometry.PointCloud:
    merged_point_cloud = o3d.geometry.PointCloud()
    for point_cloud, camera_to_shared_frame_transform in zip(point_clouds, camera_to_shared_frame_transforms):
        point_cloud.transform(camera_to_shared_frame_transform)
        merged_point_cloud += point_cloud

    return merged_point_cloud

def teleop_to_goal_pose(robot: ROSRobot) -> None:
    # Do some teleoperation to move robot to goal pose
    pass

def collect_demonstration(args: argparse.Namespace) -> None:
    # Set up interfaces/communication with cameras, robot, etc.
        # Determine cameras for collection nist board/lego board
        # Determine camera for collecting in hand point cloud
    in_hand_cameras, board_cameras = setup_cameras()
    robot = setup_robot() 

    if args.debug:
        while True:
            in_hand_pcd = in_hand_cameras[0].get_point_cloud(
                min_mm = 250,
                max_mm = 375,
                save_points = False,
                use_new_frame = True
            )

            in_hand_points = np.array(in_hand_pcd.points)
            new_in_hand_points = []
            for point in in_hand_points:
                if point[0] > -50:
                    new_in_hand_points.append(point)

            in_hand_points = np.array(new_in_hand_points)
            in_hand_pcd.points = o3d.utility.Vector3dVector(in_hand_points)
            o3d.visualization.draw_geometries([in_hand_pcd]),

            # for board_camera in board_cameras:
            #     board_pcd = board_camera.get_point_cloud(
            #         min_mm = 400,
            #         max_mm = 1000,
            #         save_points = False,
            #         use_new_frame = False
            #     )

            #     o3d.visualization.draw_geometries([board_pcd])
    
    # Move robot arm out of way of board cameras
    robot_home_pose = {
        'translation': None,
        'rotation': None
    }
    robot.move_to_pose(
        translation=robot_home_pose['translation'],
        rotation=robot_home_pose['rotation']
    )

    # Collect data from cameras (point clouds)
    board_point_clouds = []
    for board_camera in board_cameras:
        board_pcd = board_camera.get_point_cloud(
            min_mm = 400,
            max_mm = 1000,
            save_points = False,
            use_new_frame = True
        )
        board_point_clouds.append(board_pcd)

    # Merge point clouds
    camera_to_shared_frame_transforms = [np.eye(4) for _ in board_point_clouds]
    merged_board_point_cloud = merge_point_clouds(
        board_point_clouds, 
        camera_to_shared_frame_transforms
    )

    # Optionally execute object grasp
    grasp_pose = {
        'translation': None,
        'rotation': None
    }
    robot.move_to_pose(
        translation=grasp_pose['translation'],
        rotation=grasp_pose['rotation']
    )

    # Move robot arm to position(s) for in hand data collection
    inhand_collect_poses = [
        {
            'translation': {
                'x': 0.22877801647655177, 
                'y': -0.15594224298730125, 
                'z': 0.4115711357854914
            }, 
            'rotation': {
                'x': 0.26721381654687515, 
                'y': -0.6376515645326868, 
                'z': 0.6630248499858139, 
                'w': 0.28704582699764
            }
        },
    ]
    in_hand_point_clouds = []
    for inhand_collect_pose in inhand_collect_poses:
        robot.move_to_pose(
            position=inhand_collect_pose['translation'],
            orientation=inhand_collect_pose['rotation']
        )

        # Collect data from in hand camera (point clouds)
        for in_hand_camera in in_hand_cameras:
            in_hand_pcd = in_hand_camera.get_point_cloud(
                min_mm = 250,
                max_mm = 375,
                save_points = False,
                use_new_frame = True
            )
            in_hand_point_clouds.append(in_hand_pcd)

    # Merge point clouds
    camera_to_shared_frame_transforms = [np.eye(4) for _ in in_hand_point_clouds]
    merged_in_hand_point_cloud = merge_point_clouds(
        in_hand_point_clouds, 
        camera_to_shared_frame_transforms
    )

    # Get transform from shared frame to last in hand data collection pose

    # Move arm to placement position
    teleop_to_goal_pose(robot)

    # Get pose of the robot EEF in the goal pose in the shared frame
    eef_goal_pose = {
        'translation': None,
        'rotation': None
    }

    # Calculate robot EEF transform from in hand data collection pose to placement pose
    in_hand_to_eef_goal_transform = np.eye(4)

    # Transform in hand data collection point cloud to placement pose
    merged_in_hand_point_cloud.transform(in_hand_to_eef_goal_transform)

    # Format data for digital back bone

    # Store data for digital back bone

    # Format data for taxpose training

    # Store data for taxpose training

    pass

def get_args():
    parser = argparse.ArgumentParser(description="Collect demonstration data for taxpose")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()

    collect_demonstration(args)
    
    