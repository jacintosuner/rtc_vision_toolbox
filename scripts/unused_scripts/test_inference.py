import argparse
import glob
import hydra
import numpy as np
import omegaconf
import open3d as o3d
import os
import plotly.graph_objects as go
from pytorch3d.transforms import Transform3d
import time
import torch
from typing import Dict, Tuple, List


from camera.orbbec.ob_camera import OBCamera
from robot.robotiq.robotiq_gripper import RobotiqGripper
from robot.ros_robot.ros_robot import ROSRobot
from taxpose.datasets.augmentations import maybe_downsample
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
from taxpose.utils.load_model import get_weights_path
from taxpose.utils.se3 import random_se3

INHAND_DEVICES_SN = ['CL8FC3100NM']
# INHAND_DEVICE_IDS = [2]
# BOARD_DEVICE_IDS = [0, 1]
BOARD_DEVICES_SN = ['CL8FC3100RL', 'CL8FC3100W3']

ROBOT_IP = "172.26.179.142"
ROBOT_NAME = "yk_builder"

DEMO_POSES_PATH = '/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/preprogrammed_poses/wp-grasp-place-0'
BASE2CAM_TF = {}
T = []

def setup_transforms():
    transform_files = glob.glob(f'/home/mfi/repos/rtc_vision_toolbox/data/calibration_data/T_*.npy')
    transforms = []
    for filepath in transform_files:
        print(f'filename: {filepath.split("/")[-1]}')
        transform = np.load(filepath, allow_pickle=True)
        transforms.append(transform)

    global BASE2CAM_TF
    BASE2CAM_TF = {
        'CL8FC3100RL': transforms[2], # this is T_base2cam0.npy
        'CL8FC3100W3': transforms[1], # this is T_base2cam1.npy
        'CL8FC3100NM': transforms[0], # this is T_base2cam2.npy
    }

    global T
    T = [np.array(BASE2CAM_TF['CL8FC3100RL'])]
    T.append(np.array(BASE2CAM_TF['CL8FC3100W3']))
    T.append(np.array(BASE2CAM_TF['CL8FC3100NM']))

    origin = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    T.append(origin)

def load_taxpose_model(
        model_path: str, 
        model_cfg: omegaconf.dictconfig.DictConfig, 
        wandb_cfg: omegaconf.dictconfig.DictConfig, 
        task_cfg: omegaconf.dictconfig.DictConfig,
        run: str=None
    ):
    print(f"Loading TaxPose model with config: {model_cfg}")

    ckpt_file = get_weights_path(model_path, wandb_cfg, run=run)
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

def setup_cameras() -> Tuple[Dict[str, OBCamera], Dict[str, OBCamera]]:
    # board_cameras = [OBCamera(serial_no=sn) for sn in BOARD_DEVICES_SN]
    # in_hand_cameras = [OBCamera(serial_no=sn) for sn in INHAND_DEVICES_SN]
    # board_cameras = [OBCamera(device_index=id) for id in BOARD_DEVICE_IDS]
    # in_hand_cameras = [OBCamera(device_index=id) for id in INHAND_DEVICE_IDS]

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

def filter_point_cloud_bounds(pointcloud: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]) -> np.ndarray:
    return pointcloud[
        (pointcloud[:,0] > x_range[0]) & \
        (pointcloud[:,0] < x_range[1]) & \
        (pointcloud[:,1] > y_range[0]) & \
        (pointcloud[:,1] < y_range[1]) & \
        (pointcloud[:,2] > z_range[0]) & \
        (pointcloud[:,2] < z_range[1])
    ]

def capture_board_scene_pcd(
        cameras: Dict[str, OBCamera], 
        mode: str='from_disk', 
        device: str='cuda'
    ) -> np.ndarray:

    if mode == 'from_disk':
        demo_data = np.load(
            '/home/mfi/repos/rtc_vision_toolbox/data/train_data/taxpose/nist_waterproof/place/test/0_teleport_obj_points.npz',
            allow_pickle=True
        )
        points_raw_np = demo_data['clouds']
        classes_raw_np = demo_data['classes']

        points_action_np = points_raw_np[classes_raw_np == 0].copy()
        points_action_mean_np = np.mean(points_action_np, axis=0)
        points_action_np = points_action_np - points_action_mean_np

        points_anchor_np = points_raw_np[classes_raw_np == 1].copy()
        points_anchor_np = points_anchor_np - points_action_mean_np

        points_anchor = points_anchor_np.astype(np.float32)[None, ...]

        print(f'points_anchor.shape: {points_anchor.shape}')

        points_anchor = maybe_downsample(points_anchor, 2048, 'fps')

        T1 = random_se3(
            1,
            rot_var=2*np.pi,
            trans_var=0.5,
            device=device,
            rot_sample_method='random_flat_upright'
        )

        points_anchor = torch.from_numpy(points_anchor).to(device)

        board_scene_pcd = T1.transform_points(points_anchor)
        raise NotImplementedError("Need to add part_scene_pcd to from_disk mode.")

    elif mode == 'from_cameras':
        # Capture point clouds from cameras
        point_clouds = collect_camera_point_clouds(cameras)

        # TODO: Remove this hardcode
        board_scene_pcd_o3d = point_clouds['CL8FC3100RL']

        # Preprocess point cloud
        board_scene_pcd_np = np.asarray(board_scene_pcd_o3d.points)
        board_scene_pcd_np = board_scene_pcd_np / 1000 # Convert to meters
        
        # Transform from camera frame to robot base frame
        board_scene_pcd_np_h = np.hstack((board_scene_pcd_np, np.ones((board_scene_pcd_np.shape[0], 1))))
        # TODO: Remvoe this hardcode
        board_scene_pcd_np = np.matmul(BASE2CAM_TF['CL8FC3100RL'], board_scene_pcd_np_h.T).T[:, :3]

        # Filter point cloud
        board_scene_pcd = filter_point_cloud_bounds(
            board_scene_pcd_np,
            x_range=(0.04, 0.44), 
            y_range=(-0.55, -0.15), 
            z_range=(0.1, 0.3)
        )
        part_scene_pcd = filter_point_cloud_bounds(
            board_scene_pcd_np,
            x_range=(0.04, 0.44), 
            y_range=(-0.15, 0.1), 
            z_range=(0.1, 0.3)
        )

        # Downsample
        # TODO Remove these hardcodes
        num_points = 2048
        downsample_type = 'fps'
        if board_scene_pcd.shape[0] > 2048:
            board_scene_pcd = maybe_downsample(board_scene_pcd[None, ...], num_points, downsample_type)
        if part_scene_pcd.shape[0] > 2048:
            part_scene_pcd = maybe_downsample(part_scene_pcd[None, ...], num_points, downsample_type)

        board_scene_pcd = torch.from_numpy(board_scene_pcd).to(device)
        part_scene_pcd = torch.from_numpy(part_scene_pcd).to(device)
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    return board_scene_pcd, part_scene_pcd


def capture_inhand_pcd(
        cameras: Dict[str, OBCamera],
        mode: str='from_disk', 
        device: str='cuda',
        x_range: Tuple[float, float] = (0.115, 0.415),
        y_range: Tuple[float, float] = (-0.385, -0.185),
        z_range: Tuple[float, float] = (0.345, 0.595)
    ) -> np.ndarray:

    if mode == 'from_disk':
        demo_data = np.load(
            '/home/mfi/repos/rtc_vision_toolbox/data/train_data/taxpose/nist_waterproof/place/test/0_teleport_obj_points.npz',
            allow_pickle=True
        )
        points_raw_np = demo_data['clouds']
        classes_raw_np = demo_data['classes']

        points_action_np = points_raw_np[classes_raw_np == 0].copy()
        points_action_mean_np = np.mean(points_action_np, axis=0)
        points_action_np = points_action_np - points_action_mean_np

        points_action = points_action_np.astype(np.float32)[None, ...]

        print(f'points_action.shape: {points_action.shape}')

        points_action = maybe_downsample(points_action, 2048, 'fps')

        T0 = random_se3(
            1,
            rot_var=2*np.pi,
            trans_var=0.5,
            device=device,
            rot_sample_method='quat_uniform'
        )

        points_action = torch.from_numpy(points_action).to(device)

        inhand_pcd = T0.transform_points(points_action)

    elif mode == 'from_cameras':
        # Capture point clouds from cameras
        point_clouds = collect_camera_point_clouds(cameras)

        # TODO: Remove this hardcode
        inhand_pcd_o3d = point_clouds['CL8FC3100NM']

        # Preprocess point cloud
        inhand_pcd_np = np.asarray(inhand_pcd_o3d.points)
        inhand_pcd_np = inhand_pcd_np / 1000 # Convert to meters

        # Transform from camera frame to robot base frame
        inhand_pcd_np_h = np.hstack((inhand_pcd_np, np.ones((inhand_pcd_np.shape[0], 1))))
        # TODO: Remvoe this hardcode
        inhand_pcd_np = np.matmul(BASE2CAM_TF['CL8FC3100NM'], inhand_pcd_np_h.T).T[:, :3]

        # Filter point cloud
        inhand_pcd = filter_point_cloud_bounds(
            inhand_pcd_np,
            x_range=x_range, 
            y_range=y_range, 
            z_range=z_range
        )

        # Downsample
        # TODO Remove these hardcodes
        num_points = 2048
        downsample_type = 'fps'
        if inhand_pcd.shape[0] > 2048:
            inhand_pcd = maybe_downsample(inhand_pcd[None, ...], num_points, downsample_type)

        inhand_pcd = torch.from_numpy(inhand_pcd).to(device)

    else:
        raise ValueError(f'Unsupported mode: {mode}')

    return inhand_pcd

def execute_grasp(gripper: RobotiqGripper = None, mode: str = 'manual') -> None:
    if mode == 'manual':
        input("Press enter to execute grasp...")
    elif mode == 'robotiq':
        assert gripper is not None and isinstance(gripper, RobotiqGripper), "RobotiqGripper object must be provided for robotiq mode"
        gripper.closeGripper()
    pass


def plot_multi_np(plist):
    """
    Args: plist, list of numpy arrays of shape, (1,num_points,3)
    """
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#8c564b',  # chestnut brown
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#8c564b',  # chestnut brown
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#8c564b',  # chestnut brown
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
    ]
    skip = 1
    go_data = []
    for i in range(len(plist)):
        p_dp = plist[i]
        plot = go.Scatter3d(x=p_dp[::skip,0], y=p_dp[::skip,1], z=p_dp[::skip,2], 
                     mode='markers', marker=dict(size=2, color=colors[i],
                     symbol='circle'))
        go_data.append(plot)
 
    layout = go.Layout(
        scene=dict(
            aspectmode='data',
        ),
        height=1200,
        width=900,
    )
    
    colors = ['red', 'green', 'blue']  # X, Y, Z axis colors
    
    fig = go.Figure()

    for tf in T:
        origin = tf[:3, 3]
        axes = tf[:3, :3]

        for i in range(3):
            axis_end = origin + 0.3*axes[:, i]
            fig.add_trace(go.Scatter3d(
                x=[origin[0], axis_end[0]],
                y=[origin[1], axis_end[1]],
                z=[origin[2], axis_end[2]],
                mode='lines',
                line=dict(color=colors[i], width=4),
                name='Axis ' + str(i+1)
            ))

    for plot in go_data:
        fig.add_trace(plot)

    # add axis lines and camera view
    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        camera = dict(eye=dict(x=-0.30, y=0, z=-.25),
                      center=dict(x=0., y=0, z=-0.25),
                      up=dict(x=0, y=0, z=1),
                     )
        ),
        height=800,
        width=1200,
    )
                      
    fig.show()

@hydra.main(config_path="../models/taxpose/configs", config_name="eval_mfi")
def taxpose_main(cfg):
    ##############################################################################
    # Setup
    VISUALIZE = True
    
    print(omegaconf.OmegaConf.to_yaml(cfg, resolve=True))
    print(f'cfg type: {type(cfg.place_model)}')
    place_model = load_taxpose_model(
        cfg.checkpoints.place,
        cfg.place_model,
        cfg.wandb,
        cfg.place_task,
        None
    )
    print(f'Loaded TaxPose Place model: {place_model}')
    grasp_model = load_taxpose_model(
        cfg.checkpoints.grasp,
        cfg.grasp_model,
        cfg.wandb,
        cfg.grasp_task,
        None
    )
    print(f'Loaded TaxPose Grasp model: {grasp_model}')

    # Get cameras
    in_hand_cameras, board_cameras = setup_cameras()

    # Get robot
    robot, gripper = setup_robot()
    
    # Get transforms
    setup_transforms()

    ##############################################################################

    ##########################################################################
    # Start pose
    ##########################################################################

    # Move to start pose
    start_pose = np.load(os.path.join(DEMO_POSES_PATH, "start_pose.npz")).get('arr_0')
    start_pose_translation = start_pose[:3, 3]
    start_pose_orientation = start_pose[:3, :3]
    
    res = None
    max_tries = 5
    while res == None and max_tries > 0:
        res = robot.move_to_pose(
            position=start_pose_translation,
            orientation=start_pose_orientation
        )
        max_tries -= 1

    time.sleep(0.5)

    ##########################################################################
    # Out of way pose
    ##########################################################################

    # Move to out of way pose
    out_of_way_pose = np.load(os.path.join(DEMO_POSES_PATH, "out_of_way_pose.npz")).get('arr_0')
    out_of_way_pose_translation = out_of_way_pose[:3, 3]
    out_of_way_pose_orientation = out_of_way_pose[:3, :3]
    
    res = None
    max_tries = 5
    while res == None and max_tries > 0:
        res = robot.move_to_pose(
            position=out_of_way_pose_translation,
            orientation=out_of_way_pose_orientation
        )
        max_tries -= 1

    time.sleep(0.5)

    ##########################################################################
    # Capture board pcd and part pcd
    ##########################################################################

    # TODO Capture grasp scene point cloud
    # Capture board scene point cloud
    board_scene_pcd, part_scene_pcd = capture_board_scene_pcd(
        cameras={'CL8FC3100RL': board_cameras['CL8FC3100RL']}, 
        mode='from_cameras', 
        device='cuda'
    )
    print(f'board_scene_pcd.shape: {board_scene_pcd.shape}')
    print(f'part_scene_pcd.shape: {part_scene_pcd.shape}')

    ##########################################################################
    # Gripper close up pose
    ##########################################################################

    # Move to gripper close up pose
    gripper_close_up_files = glob.glob(os.path.join(DEMO_POSES_PATH, "gripper_close_up_pose_*.npz"))
    gripper_close_up_files.sort()
    gripper_close_up_poses = [np.load(file, allow_pickle=True).get('arr_0') for file in gripper_close_up_files]
    
    # For now just do 1 inhand close up pose
    gripper_close_up_poses = [gripper_close_up_poses[0]]
    gripper_pcds = []
    for pose in gripper_close_up_poses:
        pose_translation = pose[:3, 3]
        pose_orientation = pose[:3, :3]
        
        robot.move_to_pose(
            position=pose_translation,
            orientation=pose_orientation
        )

        time.sleep(0.5)


        ##########################################################################
        # Capture gripper pcd
        ##########################################################################

        # Capture gripper point cloud
        gripper_pcd = capture_inhand_pcd(
            cameras={'CL8FC3100NM': in_hand_cameras['CL8FC3100NM']}, 
            mode='from_cameras', 
            device='cuda',
            x_range=(0.123, 0.423),
            y_range=(-0.322, -0.122),
            z_range=(0.295, 0.645)
        )
        print(f'gripper_pcd.shape: {gripper_pcd.shape}')

        gripper_pcds.append(gripper_pcd)
        
    # TODO: Don't assume only 1 inhand close up pose
    gripper_pcd = gripper_pcds[0]
    
    breakpoint()
    ##########################################################################
    # Predict grasp pose
    ##########################################################################
    
    # TODO Capture gripper close up point cloud
    # TODO Predict grasp pose
    grasp_out = grasp_model(gripper_pcd, part_scene_pcd, None, None)
    predicted_grasp_rel_transform = grasp_out['pred_T_action']
    
    if VISUALIZE:
        print(f'Visualizing Gripper, Part Scene, and Predicted Grasp Point Clouds')
        grasp_pcd = predicted_grasp_rel_transform.transform_points(gripper_pcd)
        
        gripper_pose = robot.get_eef_pose()
        gripper_pose_tf = Transform3d(
            matrix=torch.Tensor(gripper_pose.T), 
            device=predicted_grasp_rel_transform.device
        ).to(predicted_grasp_rel_transform.device)
        grasp_pose_tf = gripper_pose_tf.compose(predicted_grasp_rel_transform)
        grasp_pose = grasp_pose_tf.get_matrix().T.squeeze(-1)

        print(f'gripper_pose: \n{gripper_pose}')
        print(f'grasp_pose: \n{grasp_pose}')
        print(f'grasp_pose.shape: {grasp_pose.shape}')

        global T
        T.append(gripper_pose)
        T.append(grasp_pose.detach().cpu().numpy())
        
        plot_multi_np([
            gripper_pcd[0].detach().cpu().numpy(), 
            part_scene_pcd[0].detach().cpu().numpy(), 
            grasp_pcd[0].detach().cpu().numpy()
        ])

    state = input("Proceed with grasp? (y/n): ")
    breakpoint()
    match state:
        case "y":
            ##########################################################################
            # Execute grasp pose
            ##########################################################################
            
            print("Proceeding with grasp...")
            
            # Calculate grasp pose
            inhand_pose = robot.get_eef_pose()
            inhand_pose_tf = Transform3d(
                matrix=torch.Tensor(inhand_pose.T), 
                device=predicted_grasp_rel_transform.device
            ).to(predicted_grasp_rel_transform.device)
            grasp_pose_tf = inhand_pose_tf.compose(predicted_grasp_rel_transform)
            grasp_pose = grasp_pose_tf.get_matrix().T.squeeze(-1).detach().cpu().numpy()
            grasp_pose_translation = np.float64(grasp_pose[:3, 3])
            grasp_pose_orientation = np.float64(grasp_pose[:3, :3])
            
            # Calculate pre-grasp pose
            pre_grasp_pose_translation = grasp_pose_translation + np.array([0.0, 0.0, 0.1])
            pre_grasp_pose_orientation = grasp_pose_orientation
            
            # Move to pre-grasp pose
            robot.move_to_pose(
                position=pre_grasp_pose_translation,
                orientation=pre_grasp_pose_orientation
            )
            
            time.sleep(0.5)
            print(f"pre-grasp_pose_translation: {pre_grasp_pose_translation}")
            print(f"grasp_pose_translation: {grasp_pose_translation}")
            breakpoint()
            
            # Move to grasp pose
            robot.move_to_pose(
                position=grasp_pose_translation,
                orientation=grasp_pose_orientation
            )
            
            breakpoint()
            # Execute grasp
            gripper.closeGripper()
            
            time.sleep(0.5)
            
            # Move to post-grasp pose (same as pre-grasp pose)
            robot.move_to_pose(
                position=pre_grasp_pose_translation,
                orientation=pre_grasp_pose_orientation
            )
            
            time.sleep(0.5)
            
        case _:
            print("Terminating...")
            quit()
    
    # ##########################################################################
    # # Grasp pose
    # ##########################################################################
    
    # # TODO Move to grasp pose
    # # Optionally execute object grasp
    # grasp_pose = np.load(os.path.join(DEMO_POSES_PATH, "grasp_pose.npz")).get('arr_0')
    # grasp_pose_translation = grasp_pose[:3, 3]
    # grasp_pose_orientation = grasp_pose[:3, :3]

    # # Move to pre grasp
    # pre_grasp_pose_translation = grasp_pose_translation + np.array([0.0, 0.0, 0.1])
    # pre_grasp_pose_orientation = grasp_pose_orientation

    # while True:
    #     res = None
    #     max_tries = 5
    #     while res == None and max_tries > 0:
    #         res = robot.move_to_pose(
    #             position=pre_grasp_pose_translation,
    #             orientation=pre_grasp_pose_orientation
    #         )
    #         max_tries -= 1

    #     time.sleep(0.5)

    #     # Move to grasp
    #     res = None
    #     max_tries = 5
    #     while res == None and max_tries > 0:
    #         res = robot.move_to_pose(
    #             position=grasp_pose_translation,
    #             orientation=grasp_pose_orientation
    #         )
    #         max_tries -= 1

    #     # Execute grasp
    #     execute_grasp(gripper, mode='robotiq')

    #     time.sleep(0.5)

    #     # Move to post grasp
    #     post_grasp_pose_translation = grasp_pose_translation + np.array([0.0, 0.0, 0.1])
    #     post_grasp_pose_orientation = grasp_pose_orientation
        
    #     res = None
    #     max_tries = 5
    #     while res == None and max_tries > 0:
    #         res = robot.move_to_pose(
    #             position=post_grasp_pose_translation,
    #             orientation=post_grasp_pose_orientation
    #         )
    #         max_tries -= 1

    #     time.sleep(0.5)

    #     loop_grasp = input("Grasp again? (y/n): ")
    #     match loop_grasp:
    #         case "y":
    #             gripper.openGripper()
    #             continue
    #         case "n":
    #             break
    #         case _:
    #             print("Invalid input. Exiting...")
    #             break


    ##########################################################################
    # In-hand pose
    ##########################################################################

    # TODO Move to inhand pose
    # Move robot arm to position(s) for in hand data collection
    inhand_close_up_files = glob.glob(os.path.join(DEMO_POSES_PATH, "inhand_close_up_pose_*.npz"))
    inhand_close_up_files.sort()
    inhand_close_up_poses = [np.load(file, allow_pickle=True).get('arr_0') for file in inhand_close_up_files]
    
    # For now just do 1 inhand close up pose
    inhand_close_up_poses = [inhand_close_up_poses[0]]
    inhand_pcds = []
    for pose in inhand_close_up_poses:
        pose_translation = pose[:3, 3]
        pose_orientation = pose[:3, :3]
        
        res = None
        max_tries = 5
        while res == None and max_tries > 0:
            res = robot.move_to_pose(
                position=pose_translation,
                orientation=pose_orientation
            )
            max_tries -= 1

        time.sleep(0.5)


        ##########################################################################
        # Capture in-hand pcd
        ##########################################################################

        # Capture inhand point cloud
        inhand_pcd = capture_inhand_pcd(
            cameras={'CL8FC3100NM': in_hand_cameras['CL8FC3100NM']}, 
            mode='from_cameras', 
            device='cuda'
        )
        print(f'inhand_pcd.shape: {inhand_pcd.shape}')

        inhand_pcds.append(inhand_pcd)
        
    # TODO: Don't assume only 1 inhand close up pose
    inhand_pcd = inhand_pcds[0]

    # if VISUALIZE:
    #     print(f'Visualizing In-Hand and Board Scene Point Clouds')
    #     plot_multi_np([
    #         inhand_pcd[0].detach().cpu().numpy(), 
    #         board_scene_pcd[0].detach().cpu().numpy()
    #     ])

    ##########################################################################
    # Predict place pose
    ##########################################################################

    # Predict inhand to place pose transformation
    place_out = place_model(inhand_pcd, board_scene_pcd, None, None)
    predicted_place_rel_transform = place_out['pred_T_action']
    print(f'predicted_place_rel_transform: {predicted_place_rel_transform}')

    if VISUALIZE:
        print(f'Visualizing In-Hand, Board Scene, and Predicted Place Point Clouds')
        place_pcd = predicted_place_rel_transform.transform_points(inhand_pcd)
        
        inhand_pose = robot.get_eef_pose()
        inhand_pose_tf = Transform3d(
            matrix=torch.Tensor(inhand_pose.T), 
            device=predicted_place_rel_transform.device
        ).to(predicted_place_rel_transform.device)
        print(f'inhand_pose device: {inhand_pose_tf.device}')
        print(f'predicted_place_rel_transform device: {predicted_place_rel_transform.device}')
        place_pose_tf = inhand_pose_tf.compose(predicted_place_rel_transform)
        place_pose = place_pose_tf.get_matrix().T.squeeze(-1)

        print(f'inhand_pose: \n{inhand_pose}')
        print(f'place_pose: \n{place_pose}')
        print(f'place_pose.shape: {place_pose.shape}')

        T.append(inhand_pose)
        T.append(place_pose.detach().cpu().numpy())
        
        plot_multi_np([
            inhand_pcd[0].detach().cpu().numpy(), 
            board_scene_pcd[0].detach().cpu().numpy(), 
            place_pcd[0].detach().cpu().numpy()
        ])

        from datetime import datetime
        np.savez(f'/home/mfi/repos/rtc_vision_toolbox/data/wp-grasp-place-0-predictions/{datetime.now().strftime("%Y%m%d-%H%M%S")}_predictions.npz', 
            inhand_pcd=inhand_pcd[0].detach().cpu().numpy(), 
            board_scene_pcd=board_scene_pcd[0].detach().cpu().numpy(), 
            place_pcd=place_pcd[0].detach().cpu().numpy(),
            place_pose=place_pose.detach().cpu().numpy()
        )

    state = input("Proceed with place? (y/n): ")
    breakpoint()
    match state:
        case "y":
            ##########################################################################
            # Execute place pose
            ##########################################################################
            
            print("Proceeding with place...")
            
            # Calculate place pose
            inhand_pose = robot.get_eef_pose()
            inhand_pose_tf = Transform3d(
                matrix=torch.Tensor(inhand_pose.T), 
                device=predicted_place_rel_transform.device
            ).to(predicted_place_rel_transform.device)
            place_pose_tf = inhand_pose_tf.compose(predicted_place_rel_transform)
            place_pose = place_pose_tf.get_matrix().T.squeeze(-1).detach().cpu().numpy()
            place_pose_translation = place_pose[:3, 3]
            place_pose_orientation = place_pose[:3, :3]
            
            # Calculate pre-place pose
            pre_place_pose_translation = place_pose_translation + np.array([0.0, 0.0, 0.1])
            pre_place_pose_orientation = place_pose_orientation
            
            # Move to pre-place pose
            res = None
            max_tries = 5
            while res == None and max_tries > 0:
                res = robot.move_to_pose(
                    position=pre_place_pose_translation,
                    orientation=pre_place_pose_orientation
                )
                max_tries -= 1
            
            time.sleep(0.5)
            
            print(f'Skipping place execution for now...')
            # # Move to place pose
            # robot.move_to_pose(
            #     position=place_pose_translation,
            #     orientation=place_pose_orientation
            # )
            
            # Execute place
            # gripper.openGripper()
            input("Press enter to execute place...")            
            
            time.sleep(0.5)
            
            # # Move to post-place pose (same as pre-place pose)
            # robot.move_to_pose(
            #     position=pre_place_pose_translation,
            #     orientation=pre_place_pose_orientation
            # )
            
            time.sleep(0.5)
            
        case _:
            print("Moving to start pose and exiting...")
            pass

    # Move back to start pose
    res = None
    max_tries = 5
    while res == None and max_tries > 0:
        res = robot.move_to_pose(
            position=start_pose_translation,
            orientation=start_pose_orientation
        )
        max_tries -= 1
    gripper.openGripper()

if __name__ == "__main__":
    taxpose_main()