from calibration.calibrations import *
from calibration.marker.aruco_marker import ArucoMarker
from camera.orbbec.ob_camera import OBCamera
from robot.ros_robot.ros_robot import ROSRobot

'''
0 - CL8FC3100RL
1 - CL8FC3100W3
2 - CL8FC3100NM
'''

def test_robot_camera_calibration():
    camera = OBCamera(serial_no="CL8FC3100RL")
    
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

def test_camera_marker_calibration():
    camera = OBCamera(serial_no="CL8FC3100W3")
    
    print("=====================================")
    print("CAMERA INITIALIZED")
    print("=====================================")
    
    marker = ArucoMarker()
    
    print("=====================================")
    print("MARKER INITIALIZED")
    print("=====================================")
    
    T_marker_camera = get_camera_marker_tf(camera, marker)
    
    print("=====================================")
    print("RESULTS ARE HERE!!")
    print("=====================================")
    
    print("T_marker_camera:\n", T_marker_camera)    

if __name__ == "__main__":
    #test_camera_marker_calibration()
    test_robot_camera_calibration()
