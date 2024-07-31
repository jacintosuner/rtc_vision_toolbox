import argparse
import glob
import numpy as np
import open3d as o3d
import os
import time
import datetime
from typing import List, Tuple, Dict

from camera.orbbec.ob_camera import OBCamera
from robot.robotiq.robotiq_gripper import RobotiqGripper
from robot.ros_robot.ros_robot import ROSRobot

INHAND_DEVICES_SN = ['CL8FC3100NM']
# INHAND_DEVICE_IDS = [2]
# BOARD_DEVICE_IDS = [0, 1]
BOARD_DEVICES_SN = ['CL8FC3100RL', 'CL8FC3100W3']

ROBOT_IP = "172.26.179.142"
ROBOT_NAME = "yk_builder"

def setup_cameras() -> Tuple[Dict[str, OBCamera], Dict[str, OBCamera]]:
    in_hand_cameras = {}
    for sn in INHAND_DEVICES_SN:
        in_hand_cameras[sn] = OBCamera(serial_no=sn)
    
    board_cameras = {}
    for sn in BOARD_DEVICES_SN:
        board_cameras[sn] = OBCamera(serial_no=sn)

    return in_hand_cameras, board_cameras

def setup_robot() -> Tuple[ROSRobot, RobotiqGripper]:
    robot = ROSRobot(
        rosmaster_ip=ROBOT_IP, 
        robot_name=ROBOT_NAME
    )
    gripper = RobotiqGripper("/dev/ttyUSB0")
    return robot, gripper

def merge_point_clouds(
        point_clouds: List[o3d.geometry.PointCloud], 
        camera_to_shared_frame_transforms: List[np.ndarray]
    ) -> o3d.geometry.PointCloud:
    merged_point_cloud = o3d.geometry.PointCloud()
    for point_cloud, camera_to_shared_frame_transform in zip(point_clouds, camera_to_shared_frame_transforms):
        point_cloud.transform(camera_to_shared_frame_transform)
        merged_point_cloud += point_cloud

    return merged_point_cloud

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

def execute_grasp(gripper: RobotiqGripper = None, mode: str = 'manual') -> None:
    if mode == 'manual':
        input("Press enter to execute grasp...")
    elif mode == 'robotiq':
        assert gripper is not None and isinstance(gripper, RobotiqGripper), "RobotiqGripper object must be provided for robotiq mode"
        gripper.closeGripper()
    pass

def collect_demonstration(args: argparse.Namespace) -> None:
    # Set up interfaces/communication with cameras, robot, etc.
        # Determine cameras for collection nist board/lego board
        # Determine camera for collecting in hand point cloud
    in_hand_cameras, board_cameras = setup_cameras()
    robot, gripper = setup_robot() 

    demo_path = os.path.join(args.demo_save_path, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(demo_path, exist_ok=True)

    if args.debug:
        while True:
            print(f'collecting from inhand camera')
            in_hand_pcd = in_hand_cameras[INHAND_DEVICES_SN[0]].get_point_cloud(
                min_mm = 0,
                max_mm = 500,
                save_points = False,
                use_new_frame = True
            )

            if in_hand_pcd is not None:

                in_hand_points = np.array(in_hand_pcd.points)
                # new_in_hand_points = []
                # for point in in_hand_points:
                #     if point[0] > -50:
                #         new_in_hand_points.append(point)

                # in_hand_points = np.array(new_in_hand_points)
                in_hand_pcd.points = o3d.utility.Vector3dVector(in_hand_points)
                o3d.visualization.draw_geometries([in_hand_pcd]),

            for key, board_camera in board_cameras.items():
                print(f'collecting from camera: {board_camera}')
                board_pcd = board_camera.get_point_cloud(
                    min_mm = 0,
                    max_mm = 5000,
                    save_points = False,
                    use_new_frame = True
                )

                if board_pcd is not None:
                    o3d.visualization.draw_geometries([board_pcd])

            input('continue')

    ##########################################################################
    # Start pose
    ##########################################################################

    print(f'Starting...')
    breakpoint()

    # Move to start pose
    start_pose = np.load(os.path.join(args.demo_poses_path, "start_pose.npz")).get('arr_0')
    start_pose_translation = start_pose[:3, 3]
    start_pose_orientation = start_pose[:3, :3]
    robot.move_to_pose(
        position=start_pose_translation,
        orientation=start_pose_orientation
    )

    time.sleep(0.5)

    print(f'start pose: \n{start_pose}')
    breakpoint()

    ##########################################################################
    # Out of way pose
    ##########################################################################

    # Move to out of way pose
    out_of_way_pose = np.load(os.path.join(args.demo_poses_path, "out_of_way_pose.npz")).get('arr_0')
    out_of_way_pose_translation = out_of_way_pose[:3, 3]
    out_of_way_pose_orientation = out_of_way_pose[:3, :3]
    robot.move_to_pose(
        position=out_of_way_pose_translation,
        orientation=out_of_way_pose_orientation
    )

    time.sleep(0.5)

    # Collect data from cameras (point clouds)
    out_of_way_pcd_dict = collect_camera_point_clouds({**board_cameras, **in_hand_cameras})

    # Move to start pose
    start_pose = np.load(os.path.join(args.demo_poses_path, "start_pose.npz")).get('arr_0')
    start_pose_translation = start_pose[:3, 3]
    start_pose_orientation = start_pose[:3, :3]
    robot.move_to_pose(
        position=start_pose_translation,
        orientation=start_pose_orientation
    )
    
    time.sleep(0.5)

    print(f'out of way pose: \n{out_of_way_pose}')
    breakpoint()

    ##########################################################################
    # Gripper close up pose
    ##########################################################################

    # Move robot arm to position(s) for gripper close up data collection
    gripper_close_up_files = glob.glob(os.path.join(args.demo_poses_path, "gripper_close_up_pose_*.npz"))
    gripper_close_up_files.sort()
    gripper_close_up_poses = [np.load(file, allow_pickle=True).get('arr_0') for file in gripper_close_up_files]
    gripper_pcd_dicts = []
    for pose in gripper_close_up_poses:
        pose_translation = pose[:3, 3]
        pose_orientation = pose[:3, :3]
        
        robot.move_to_pose(
            position=pose_translation,
            orientation=pose_orientation
        )

        time.sleep(0.5)

        # Collect data from cameras (point clouds)
        gripper_pcd_dict = collect_camera_point_clouds(in_hand_cameras)
        gripper_pcd_dicts.append(gripper_pcd_dict)

    print(f'Gripper close up poses: ')
    for pose in gripper_close_up_poses:
        print(f'\t{pose}')
    breakpoint()
    ##########################################################################
    # Grasp pose
    ##########################################################################

    # Optionally execute object grasp
    grasp_pose = np.load(os.path.join(args.demo_poses_path, "grasp_pose.npz")).get('arr_0')
    grasp_pose_translation = grasp_pose[:3, 3]
    grasp_pose_orientation = grasp_pose[:3, :3]

    # Move to pre grasp
    pre_grasp_pose_translation = grasp_pose_translation + np.array([0.0, 0.0, 0.1])
    pre_grasp_pose_orientation = grasp_pose_orientation

    while True:
        robot.move_to_pose(
            position=pre_grasp_pose_translation,
            orientation=pre_grasp_pose_orientation
        )

        time.sleep(0.5)

        # Move to grasp
        robot.move_to_pose(
            position=grasp_pose_translation,
            orientation=grasp_pose_orientation
        )

        execute_grasp(gripper, mode='robotiq')

        time.sleep(0.5)

        # Move to post grasp
        post_grasp_pose_translation = grasp_pose_translation + np.array([0.0, 0.0, 0.1])
        post_grasp_pose_orientation = grasp_pose_orientation
        robot.move_to_pose(
            position=post_grasp_pose_translation,
            orientation=post_grasp_pose_orientation
        )

        time.sleep(0.5)

        loop_grasp = input("Grasp again? (y/n): ")
        match loop_grasp:
            case "y":
                gripper.openGripper()
                continue
            case "n":
                break
            case _:
                print("Invalid input. Exiting...")
                break

    print(f'Grasp pose: \n{grasp_pose}')
    breakpoint()
    ##########################################################################
    # In-hand pose
    ##########################################################################

    # Move robot arm to position(s) for in hand data collection
    inhand_collect_pose_files = glob.glob(os.path.join(args.demo_poses_path, "inhand_close_up_pose_*.npz"))
    inhand_collect_poses = [np.load(file, allow_pickle=True).get('arr_0') for file in inhand_collect_pose_files]
    in_hand_pcd_dicts = []
    for pose in inhand_collect_poses:
        pose_translation = pose[:3, 3]
        pose_orientation = pose[:3, :3]
        
        robot.move_to_pose(
            position=pose_translation,
            orientation=pose_orientation
        )

        time.sleep(0.5)

        # Collect data from cameras (point clouds)
        in_hand_pcd_dict = collect_camera_point_clouds(in_hand_cameras)
        in_hand_pcd_dicts.append(in_hand_pcd_dict)

    print(f'In-hand close up poses: ')
    for pose in inhand_collect_poses:
        print(f'\t{pose}')
    breakpoint()
    ##########################################################################
    # Placement pose
    ##########################################################################

    # Go home first
    robot.move_to_pose(
        position=start_pose_translation,
        orientation=start_pose_orientation
    )

    time.sleep(0.5)

    # Get pose of the robot EEF in the goal pose in the shared frame
    eef_goal_pose = np.load(os.path.join(args.demo_poses_path, "placement_pose.npz")).get('arr_0')
    eef_goal_pose_translation = eef_goal_pose[:3, 3]
    eef_goal_pose_orientation = eef_goal_pose[:3, :3]

    # Move to pre placement
    pre_placement_pose_translation = eef_goal_pose_translation + np.array([0.0, 0.0, 0.03])
    pre_placement_pose_orientation = eef_goal_pose_orientation
    robot.move_to_pose(
        position=pre_placement_pose_translation,
        orientation=pre_placement_pose_orientation
    )

    time.sleep(0.5)

    # Move to placement
    robot.move_to_pose(
        position=eef_goal_pose_translation,
        orientation=eef_goal_pose_orientation
    )

    # Collect data from cameras (point clouds)
    placement_pcd_dict = collect_camera_point_clouds({**board_cameras, **in_hand_cameras})

    time.sleep(0.5)

    # Move to post placement
    post_placement_pose_translation = eef_goal_pose_translation + np.array([0.0, 0.0, 0.03])
    post_placement_pose_orientation = eef_goal_pose_orientation
    robot.move_to_pose(
        position=post_placement_pose_translation,
        orientation=post_placement_pose_orientation
    )

    time.sleep(0.5)

    # Move back to start pose
    robot.move_to_pose(
        position=start_pose_translation,
        orientation=start_pose_orientation
    )

    print(f'Placement pose: \n{eef_goal_pose}')

    ##########################################################################
    # Logging
    ##########################################################################

    # Save point clouds as .npz files
    demo_path = os.path.join(args.demo_save_path, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(demo_path, exist_ok=True)
    for key, pcd in out_of_way_pcd_dict.items():
        np.savez(os.path.join(demo_path, f"{key}_out_of_way.npz"), np.array(pcd.points))
    for i, gripper_pcd_dict in enumerate(gripper_pcd_dicts):
        for key, pcd in gripper_pcd_dict.items():
            np.savez(os.path.join(demo_path, f"{key}_gripper_{i}.npz"), np.array(pcd.points))
    for i, in_hand_pcd_dict in enumerate(in_hand_pcd_dicts):
        for key, pcd in in_hand_pcd_dict.items():
            np.savez(os.path.join(demo_path, f"{key}_in_hand_{i}.npz"), np.array(pcd.points))
    for key, pcd in placement_pcd_dict.items():
        np.savez(os.path.join(demo_path, f"{key}_placement.npz"), np.array(pcd.points))

    # # Format data for digital back bone

    # # Store data for digital back bone

    # # Format data for taxpose training

    # # Store data for taxpose training

    # pass

def save_demo_poses(args):
    in_hand_cameras, board_cameras = setup_cameras()
    robot, gripper = setup_robot() 

    state = input("Ready to collect demonstration poses? (y/n): ")
    match state:
        case "y":
            #########################################################
            # Start Pose
            #########################################################
            print(f'Saving poses to {args.demo_poses_path}...')
            os.makedirs(args.demo_poses_path, exist_ok=True)
            state = input("Overwrite start pose with current pose? (y/n): ")
            match state:
                case "y":
                    start_pose = robot.get_eef_pose()
                    print("Start pose: ", start_pose)
                    np.savez(os.path.join(args.demo_poses_path, "start_pose.npz"), start_pose)
                case "n":
                    print("Continuing...")

            #########################################################
            # Out of the way pose
            #########################################################
            state = input("Overwrite out of way pose with current pose? (y/n): ")
            match state:
                case "y":
                    out_of_way_pose = robot.get_eef_pose()
                    print("Out of way pose: ", out_of_way_pose)
                    np.savez(os.path.join(args.demo_poses_path, "out_of_way_pose.npz"), out_of_way_pose)
                case "n":
                    print("Continuing...")

            #########################################################
            # Gripper close up pose
            #########################################################
            state = input("Overwrite gripper close up pose with current pose? (y/n):")
            match state:
                case "y":
                    i = 0
                    while True:
                        print(f'collecting from inhand camera: {INHAND_DEVICES_SN[0]}')
                        in_hand_pcd = in_hand_cameras[INHAND_DEVICES_SN[0]].get_point_cloud(
                            min_mm = 0,
                            max_mm = 1000,
                            save_points = False,
                            use_new_frame = True
                        )

                        if in_hand_pcd is not None:
                            in_hand_points = np.array(in_hand_pcd.points)
                            in_hand_pcd.points = o3d.utility.Vector3dVector(in_hand_points)
                            o3d.visualization.draw_geometries([in_hand_pcd])

                        close_up_pose = robot.get_eef_pose()
                        print("Close up pose: ", close_up_pose)
                        np.savez(os.path.join(args.demo_poses_path, f"gripper_close_up_pose_{i}.npz"), close_up_pose)

                        redo_close_up = input("Capture another gripper close up pose? (y/n/r), `r` for redo: ")
                        match redo_close_up:
                            case "y":
                                i += 1                                
                                continue
                            case "n":
                                break
                            case "r":
                                continue
                            case _:
                                print("Invalid input. Exiting...")
                                break
                    
                case "n":
                    print("Continuing...")

            #########################################################
            # Grasp pose
            #########################################################
            state = input("Overwrite grasp pose with current pose? (y/n): ")
            match state:
                case "y":
                    grasp_pose = robot.get_eef_pose()
                    print("Grasp pose: ", grasp_pose)
                    np.savez(os.path.join(args.demo_poses_path, "grasp_pose.npz"), grasp_pose)
                case "n":
                    print("Continuing...")

            #########################################################
            # In-hand close up pose
            #########################################################
            state = input("Overwrite In-hand close up pose with current pose? (y/n):")
            match state:
                case "y":
                    i = 0
                    while True:
                        print(f'collecting from inhand camera: {INHAND_DEVICES_SN[0]}')
                        in_hand_pcd = in_hand_cameras[INHAND_DEVICES_SN[0]].get_point_cloud(
                            min_mm = 0,
                            max_mm = 1000,
                            save_points = False,
                            use_new_frame = True
                        )

                        if in_hand_pcd is not None:
                            in_hand_points = np.array(in_hand_pcd.points)
                            in_hand_pcd.points = o3d.utility.Vector3dVector(in_hand_points)
                            o3d.visualization.draw_geometries([in_hand_pcd])

                        close_up_pose = robot.get_eef_pose()
                        print("Close up pose: ", close_up_pose)
                        np.savez(os.path.join(args.demo_poses_path, f"inhand_close_up_pose_{i}.npz"), close_up_pose)

                        redo_close_up = input("Capture another close up pose? (y/n/r), `r` for redo: ")
                        match redo_close_up:
                            case "y":
                                i += 1                                
                                continue
                            case "n":
                                break
                            case "r":
                                continue
                            case _:
                                print("Invalid input. Exiting...")
                                break
                    
                case "n":
                    print("Continuing...")

            #########################################################
            # Place pose
            #########################################################
            state = input("Overwrite placement pose with current pose? (y/n):")
            match state:
                case "y":
                    placement_pose = robot.get_eef_pose()
                    print("Placement pose: ", placement_pose)
                    np.savez(os.path.join(args.demo_poses_path, "placement_pose.npz"), placement_pose)
                case "n":
                    print("Continuing...")

        case "n":
            print("Exiting...")
        case _:
            print("Invalid input. Exiting...")

def get_args():
    parser = argparse.ArgumentParser(description="Collect demonstration data for taxpose")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--save_full_demo", action="store_true", help="Run in demonstration mode")
    parser.add_argument("--save_demo_poses", action="store_true", help="Run in demonstration mode")
    parser.add_argument("--demo_poses_path", type=str, help="Path to save demonstration poses")
    parser.add_argument("--demo_save_path", type=str, help="Path to save demonstration data")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()

    if args.save_full_demo:
        collect_demonstration(args)
    elif args.save_demo_poses:
        save_demo_poses(args)
    