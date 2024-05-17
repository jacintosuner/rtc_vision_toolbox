import argparse
from calibration.calibrations import *
from calibration.marker.aruco_marker import ArucoMarker
from camera.orbbec.ob_camera import OBCamera
from camera.zed_ros.zed_ros import ZedRos
from robot.ros_robot.ros_robot import ROSRobot


cam_key = { 'cam0': 'CL8FC3100RL',
            'cam1': 'CL8FC3100W3',
            'cam2': 'CL8FC3100NM'}

def robot_camera_calibration(args):
    
    # get input if the specific args are not provided
    
    if args.cam_class is None:
        camera_class = input("Choose which class of camera. \n1:orbbec, \n2:zed_ros\n(1/2): ")
    if args.cam_id is None:
        camera_id = input("Choose which camera to use. \n0:cam0, \n1:cam1, \n2:cam2\n(0/1/2): ")
    
    match camera_class:
        case "1":
            camera = OBCamera(serial_no=cam_key[f'cam{camera_id}'])
        case "2":
            if camera_id == 2:
                camera = ZedRos(camera_node=f'/cam{camera_id}/zed_cam{camera_id}', 
                                camera_type='zedxm')
            else:
                camera = ZedRos(camera_node=f'/cam{camera_id}/zed_cam{camera_id}', 
                                camera_type='zedx')
        case _:
            print("Invalid choice")
            exit()
    
    print("=====================================")
    print("CAMERA INITIALIZED")
    print("=====================================")
    
    marker = ArucoMarker(size=0.05)
    
    print("=====================================")
    print("MARKER INITIALIZED")
    print("=====================================")
    
    robot = ROSRobot(robot_name='yk_builder', rosmaster_ip='172.26.179.142')

    print("=====================================")
    print("ROBOT INITIALIZED")
    print("=====================================")

    T_robot2camera = get_robot_camera_tf(camera, robot, marker, 'PLAY', use_depth=False)
    
    print("=====================================")
    print("RESULTS ARE HERE!!")
    print("=====================================")
    
    print("T_robot2camera:\n", T_robot2camera)

def parse_args():
    parser = argparse.ArgumentParser(description='Robot Camera Calibration')
    parser.add_argument('--cam_id', type=str, help='Camera ID (cam0, cam1, cam2)')
    parser.add_argument('--cam_class', type=str, help='Camera Class (orbbec, zed_ros)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    robot_camera_calibration(args)