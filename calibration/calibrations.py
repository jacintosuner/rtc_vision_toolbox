import numpy as np
from datetime import datetime
#from autolab_core import RigidTransform
import cv2
import time

def get_camera_marker_tf(camera, marker):
    
    image = camera.get_rgb_image()
    cv2.imwrite("test.png",image)
    # image = camera.get_ir_image()
    
    # get camera intrinsics
    camera_matrix = camera.get_rgb_intrinsics()
    print(camera_matrix.astype(np.float32))
    camera_distortion = camera.get_rgb_distortion()
    
    # get aruco marker poses w.r.t. camera
    transforms, ids = marker.get_center_poses(image, camera_matrix, camera_distortion, debug=True)
    
    # print the transformation
    for i in range(len(ids)):
        print("Transform for id{}:\n {}".format(ids[i], transforms[i]))
        
    return transforms[0]
        
def get_robot_camera_tf(camera, robot, marker, num_trials=None, verbose=True):
    
    # 1. Initialize the robot, camera and marker
    T_marker_camera_set = []
    T_eef_base_set = []

    # 2. Collect data
    if num_trials is None:
        num_trials = int(input("Enter the number of trials: "))

    if num_trials < 2 or num_trials > 10:
        print("Invalid number of trials. Aborting...")
        return
        
    home_pose = robot.get_eef_pose()
    for i in range(num_trials):
        # Safety Check
        go_on = input("Ensure the robot is in safe position. Do you want to continue? (y/n): ")
        if(go_on == 'n'):
            break
        
        # 2.1. Collect data from camera
        image = camera.get_rgb_image()
        camera_matrix = camera.get_rgb_intrinsics()
        camera_distortion = camera.get_rgb_distortion()
        transforms, ids = marker.get_center_poses(image, camera_matrix, camera_distortion)
        
        if ids is None:
            print("No markers found. Skipping this trial.")
            continue
        else:
            T_marker_camera_set.append(transforms[0])
            if verbose:
                print(f"Marker pose in camera frame:\n {transforms[0]}")
            
            # 2.2. Collect data from robot
            gripper_pose = robot.get_eef_pose()
            T_eef_base_set.append(gripper_pose)
            if verbose:
                print(f"Gripper pose in base frame:\n {gripper_pose}")
        
        # 2.3. Move the robot with controller and collect data
        good_to_go = 'n'
        while good_to_go != 'y':
            if i == num_trials - 1:
                good_to_go = 'y'
            else:
                good_to_go = input("Jog the robot. Done? (y/n): ")
        
        # 2.3. TODO: Move the robot and collect data
        # random_delta_pos = [np.random.uniform(-0.08, 0.08, size=(3,))]
        # random_delta_axis_angle = [np.random.uniform(-0.6, 0.6, size=(3,))]
        # #new_pose = home_pos + random_delta_pos + random_delta_axis_angle
        # robot.move_to_pose(position=home_pose.position + random_delta_pos, 
        #                    orientation=home_pose.orientation + random_delta_axis_angle)
    
    # 2.4. Save the data with data and timestamp for debugging
    print(datetime.now().strftime("%Y%m%d%H%M"))
    now = datetime.now().strftime("%Y%m%d%H%M")
    np.save("T_marker_camera_set"+ now +".npy", T_marker_camera_set)
    np.save("T_eef_base_set"+ now +".npy", T_eef_base_set)
    
    # 3. Solve for the transformation
    # 3.1. Solve the extrinsic calibration between the marker and the base
    
    T_marker_eef = np.array([
        [1.0, 0.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0]
    ])

    T_marker_base_set = [np.dot(T_marker_eef, T_eef_base) for T_eef_base in T_eef_base_set]

    T_camera_base = solve_rigid_transformation(T_marker_base_set, T_marker_camera_set)
    
    avg_error = calculate_reprojection_error(T_marker_base_set, T_eef_base_set, T_camera_base)
    
    print(f"Transformation matrix T:\n{T_camera_base}")
    print(f"Avg. reprojection error: {avg_error}")
    
    # 3.2. Save the calibration and error for debugging
    now = datetime.now().strftime("%Y%m%d%H%M")
    np.save("T_camera_base_"+ now +".npy", T_camera_base)
    
    return T_camera_base


def solve_rigid_transformation(T_marker_base_set, T_marker_camera_set):
    """
    
    Calibration Options: (Reference links below) 
    CALIB_HAND_EYE_TSAI
    CALIB_HAND_EYE_PARK 
    CALIB_HAND_EYE_HORAUD 
    CALIB_HAND_EYE_ANDREFF 
    CALIB_HAND_EYE_DANIILIDIS 
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7904a801b99
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    """
    
    T_base_marker_set = [np.linalg.inv(T) for T in T_marker_base_set]
    R_base2marker = [T[:3,:3] for T in T_base_marker_set]
    t_base2marker = [T[:3,3] for T in T_base_marker_set]
    
    R_marker2camera = [T[:3,:3] for T in T_marker_camera_set]
    t_marker2camera = [T[:3,3] for T in T_marker_camera_set]
    
    R_cam2base, t_cam2base = cv2.calibrateHandEye(R_base2marker, t_base2marker, 
                                                  R_marker2camera, t_marker2camera, 
                                                  method=cv2.CALIB_HAND_EYE_TSAI)

    T = np.eye(4)
    T[:3,:3] = R_cam2base    
    T[:3,3] = np.squeeze(t_cam2base)
    
    return T

    
def calculate_reprojection_error(tag_poses, target_poses, T_matrix):
    errors = []
    for tag_pose, target_pose in zip(tag_poses, target_poses):
        # Transform target pose using T_matrix
        transformed_target = np.dot(T_matrix, target_pose)
        transformed_pos = transformed_target[:3, 3]

        # Compare with tag pos
        tag_pos = tag_pose[:3, 3]
        error = np.linalg.norm(tag_pos - transformed_pos)
        errors.append(error)

    # Compute average error
    avg_error = np.mean(errors)
    return avg_error

'''
def solve_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    
    print("inpts: ", inpts)
    print("outpts: ", outpts)
    
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    
    outpts -= outpt_mean
    inpts -= inpt_mean
    
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean)
    
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    
    return T
'''