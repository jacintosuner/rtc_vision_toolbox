import argparse
from calibration.calibrations import *
from calibration.marker.aruco_marker import ArucoMarker
from camera.orbbec.ob_camera import OBCamera
from robot.ros_robot.ros_robot import ROSRobot


cam_key = { 'cam0': 'CL8FC3100RL',
            'cam1': 'CL8FC3100W3',
            'cam2': 'CL8FC3100NM'}

def robot_camera_calibration(serial_no):
    camera = OBCamera(serial_no=serial_no)
    
    print("=====================================")
    print("CAMERA INITIALIZED")
    print("=====================================")
    
    marker = ArucoMarker()
    
    print("=====================================")
    print("MARKER INITIALIZED")
    print("=====================================")
    
    robot = ROSRobot(robot_name='yk_builder', rosmaster_ip='172.26.179.142')

    print("=====================================")
    print("ROBOT INITIALIZED")
    print("=====================================")

    T_robot2camera = get_robot_camera_tf(camera, robot, marker, 'PLAY')
    
    print("=====================================")
    print("RESULTS ARE HERE!!")
    print("=====================================")
    
    print("T_robot2camera:\n", T_robot2camera)

def parse_args():
    parser = argparse.ArgumentParser(description='Robot Camera Calibration')
    parser.add_argument('--cam_id', type=str, help='Camera ID (cam0, cam1, cam2)')
    parser.add_argument('--serial_no', type=str, help='Camera Serial Number')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.serial_no is not None:
        robot_camera_calibration(args.serial_no)
    elif args.cam_id is not None:
        serial_no = cam_key[args.cam_id]
        robot_camera_calibration(serial_no)
    else:
        print("Please provide either camera serial number or camera id")
        print("Use --help for more information")