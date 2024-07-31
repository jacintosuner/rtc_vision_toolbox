import argparse
import os
import cv2

from calibration.calibrations import *
from calibration.marker.aruco_marker import ArucoMarker
from camera.orbbec.ob_camera import OBCamera
from camera.rs_ros.rs_ros import RsRos
from camera.zed_ros.zed_ros import ZedRos
from robot.ros_robot.ros_robot import ROSRobot

cam_key = { 'cam0': 'CL8FC3100RL',
            'cam1': 'CL8FC3100W3',
            'cam2': 'CL8FC3100NM'}

def robot_camera_calibration():

    # get input if the specific args are not provided

    print("=====================================")
    print("Choose Camera params for determining marker pose")
    print("=====================================")
    camera_class = input("camera type. \n1:orbbec, \n2:zed_ros\n(1/2): ")
    camera_id = input("camera id. \n0:cam0, \n1:cam1, \n2:cam2\n(0/1/2): ")

    match camera_class:
        case "1":
            scene_camera = OBCamera(serial_no=cam_key[f"cam{camera_id}"])
        case "2":
            if camera_id == 2:
                scene_camera = ZedRos(camera_node=f'/cam{camera_id}/zed_cam{camera_id}', 
                                camera_type='zedxm')
            else:
                scene_camera = ZedRos(camera_node=f'/cam{camera_id}/zed_cam{camera_id}', 
                                camera_type='zedx')
        case _:
            print("Invalid choice")
            exit()

    inhand_camera = RsRos(camera_node='/camera/realsense2_camera', camera_type='d405')

    print("=====================================")
    print("CAMERAS INITIALIZED")
    print("=====================================")

    marker = ArucoMarker(size=0.1)

    print("=====================================")
    print("MARKER INITIALIZED")
    print("=====================================")

    robot = ROSRobot(robot_name='yk_builder', rosmaster_ip='172.26.179.142')

    print("=====================================")
    print("ROBOT INITIALIZED")
    print("=====================================")

    scene_image = scene_camera.get_rgb_image()
    scene_camera_matrix = scene_camera.get_rgb_intrinsics()
    scene_camera_distortion = scene_camera.get_rgb_distortion()
    T_scene_camera2marker, ids = marker.get_center_poses(scene_image, scene_camera_matrix, scene_camera_distortion, debug=True)
    T_scene_camera2marker = np.asarray(T_scene_camera2marker[0])

    ## get scene camera to robot transformation
    if camera_class == '1':
        file_path = os.path.join(os.path.dirname(__file__), '../data/calibration_data/femto', f'T_base2cam{camera_id}.npy')
    elif camera_class == '2':
        file_path = os.path.join(os.path.dirname(__file__), '../data/calibration_data/zed', f'T_base2cam{camera_id}.npy')

    T_base2scene_camera = np.load(file_path)
    print(f"T_scene_camera2marker:\n {T_scene_camera2marker}")
    print(f"T_base2scene_camera:\n {T_base2scene_camera}")
    T_base2marker = np.dot(T_base2scene_camera, T_scene_camera2marker)

    print(f"T_base2marker:\n {T_base2marker}")
    breakpoint()

    print("=====================================")
    print("MARKER IDENTIFIED")
    print("=====================================")

    print("Collecting data for inhand camera")

    T_base2eef_set, T_camera2marker_set = collect_data(
        camera=inhand_camera,
        robot=robot,
        marker=marker,
        method="PLAY",
        num_trials=10,
        verbose=True,
        use_depth=False,
    )
    
    now = datetime.now().strftime("%Y%m%d%H%M")
    np.save("T_camera2marker_set"+ now +".npy", T_camera2marker_set)
    np.save("T_base2eef_set"+ now +".npy", T_base2eef_set)

    print("=====================================")
    print("DATA COLLECTED")
    print("=====================================")
    
    print("Calibrating inhand camera")
    
    methods = ["ONE_SAMPLE_ESTIMATE", "SVD_ALGEBRAIC", "CALIB_HAND_EYE_TSAI", "CALIB_HAND_EYE_ANDREFF"]

    T_eef2marker_set = [np.dot(np.linalg.inv(T_base2eef), T_base2marker) for T_base2eef in T_base2eef_set]
    T_eef2camera_set = []
    avg_error_set = []
    for method in methods:
        print(f"\nMETHOD: {method}")
        T_eef2camera = solve_rigid_transformation(T_eef2marker_set, T_camera2marker_set, method=method)
        avg_error, std_error = calculate_reprojection_error(T_eef2marker_set, T_camera2marker_set, T_eef2camera)
        T_eef2camera_set.append(T_eef2camera)
        avg_error_set.append(avg_error)
        print(f"Transformation matrix T_base2camera:\n{T_eef2camera}")
        print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")

    T_eef2camera = T_eef2camera_set[np.argmin(avg_error_set)]
    np.save("T_eef2camera_"+ now +".npy", T_eef2camera)

    print("=====================================")
    print("RESULTS ARE HERE!!")
    print("=====================================")

    print("T_eef2camera:\n", T_eef2camera)

if __name__ == "__main__":
    robot_camera_calibration()
