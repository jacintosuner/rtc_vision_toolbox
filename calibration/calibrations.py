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
    '''
    Collects and solves for the transformation between the robot base and the camera
    
    Parameters:
        camera: Camera object
        robot: Robot object
        marker: Marker object
        num_trials: Number of trials to collect data
    
    Returns:
        T_base2camera: Transformation matrix from robot base to camera    
    
    Note: 
        T_a2b represents frame 'b' in frame 'a', or
        tranforms a point from frame 'b' to frame 'a'
    '''
    
    # 1. Initialize the robot, camera and marker
    T_camera2marker_set = []
    T_base2eef_set = []

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
            T_camera2marker_set.append(transforms[0])
            if verbose:
                print(f"Marker pose in camera frame:\n {transforms[0]}")
            
            # 2.2. Collect data from robot
            gripper_pose = robot.get_eef_pose()
            T_base2eef_set.append(gripper_pose)
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
    np.save("T_camera2marker_set"+ now +".npy", T_camera2marker_set)
    np.save("T_base2eef_set"+ now +".npy", T_base2eef_set)
    
    # 3. Solve for the transformation
    # 3.1. Solve the extrinsic calibration between the marker and the base
    
    T_eef2marker = np.array([
                    [1.0, 0.0, 0.0, 0.0], 
                    [0.0, 1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0, 0.0], 
                    [0.0, 0.0, 0.0, 1.0]])

    T_base2marker_set = [np.dot(T_base2eef, T_eef2marker) for T_base2eef in T_base2eef_set]
    
    T_base2camera = solve_rigid_transformation(T_base2marker_set, T_camera2marker_set)
    avg_error, std_error = calculate_reprojection_error(T_camera2marker_set, T_base2marker_set, T_base2camera)
    
    print(f"Transformation matrix T:\n{T_base2camera}")
    print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")
    
    # 3.2. Save the calibration and error for debugging
    now = datetime.now().strftime("%Y%m%d%H%M")
    np.save("T_camera_base_"+ now +".npy", T_base2camera)
    
    return T_base2camera
    
def calculate_reprojection_error(T_a2t_set, T_b2t_set, T_a2b):
    '''
    Note: 
    T_a2b represents frame 'b' in frame 'a', or
    tranforms a point from frame 'b' to frame 'a'
    '''
    errors = []
    assert len(T_a2t_set) == len(T_b2t_set)
    
    # Calculate error using transformation projections
    for i in range(len(T_b2t_set)):
        T_a2t_calc = np.dot(T_a2b, T_b2t_set[i])
        error = np.linalg.norm(T_a2t_set[i] - T_a2t_calc)
        errors.append(error)
        
    # Compute average error
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    return avg_error, std_error

def solve_rigid_transformation(T_base2target_set, T_camera2target_set, method="SVD_ALGEBRAIC"):
    """
    Solves for the rigid transformation between two sets of transformations
    
    Parameters:
        T_base2target_set: List of transformation matrices from base to target
        T_camera2target_set: List of transformation matrices from camera to target
        method: Method to use for solving the transformation
            * SVD_ALGEBRAIC: Algebraic method using SVD
            * ONE_SAMPLE_ESTIMATE: One sample estimate
            * CALIB_HAND_EYE_TSAI: Tsai's method using OpenCV library (T_base2target can be T_base2eef)
            * CALIB_HAND_EYE_ANDREFF: Andreff's method using OpenCV library (T_base2target can be T_base2eef)
            * CALIB_HAND_EYE_PARK: Park's method using OpenCV library (T_base2target can be T_base2eef)
            
    Returns:
        T_base2camera: Transformation matrix from base to camera
        
    Note:
        T_a2b represents frame 'b' in frame 'a', or
        tranforms a point from frame 'b' to frame 'a'
    """
    
    match method:
        case "SVD_ALGEBRAIC":   
            t_camera2target = np.array([T[:3,3] for T in T_camera2target_set])
            t_base2target = np.array([T[:3,3] for T in T_base2target_set])
            
            inpts = t_base2target
            outpts = t_camera2target
            
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
        
        case "ONE_SAMPLE_ESTIMATE":
            T_base2target = T_base2target_set[0]
            T_camera2target = T_camera2target_set[0]
            T_base2camera = np.dot(T_base2target, np.linalg.inv(T_camera2target))
            return T_base2camera
        
        case "CALIB_HAND_EYE_TSAI":
            T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
            R_target2base = [T[:3,:3] for T in T_target2base_set]
            t_target2base = [T[:3,3] for T in T_target2base_set]
            
            R_camera2target = [T[:3,:3] for T in T_camera2target_set]
            t_camera2target = [T[:3,3] for T in T_camera2target_set]
            R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                                R_camera2target, t_camera2target,  
                                                                cv2.CALIB_HAND_EYE_TSAI)
            T_base2camera = np.eye(4)
            T_base2camera[:3,:3] = R_camera2base
            T_base2camera[:3,3] = np.squeeze(T_base2camera)
            return T_base2camera
        
        case "CALIB_HAND_EYE_ANDREFF":
            T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
            R_target2base = [T[:3,:3] for T in T_target2base_set]
            t_target2base = [T[:3,3] for T in T_target2base_set]
            
            R_camera2target = [T[:3,:3] for T in T_camera2target_set]
            t_camera2target = [T[:3,3] for T in T_camera2target_set]
            R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                                R_camera2target, t_camera2target,  
                                                                cv2.CALIB_HAND_EYE_ANDREFF)
            T_base2camera = np.eye(4)
            T_base2camera[:3,:3] = R_camera2base
            T_base2camera[:3,3] = np.squeeze(T_base2camera)
            return T_base2camera
        
        case "CALIB_HAND_EYE_PARK":
            T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
            R_target2base = [T[:3,:3] for T in T_target2base_set]
            t_target2base = [T[:3,3] for T in T_target2base_set]
            
            R_camera2target = [T[:3,:3] for T in T_camera2target_set]
            t_camera2target = [T[:3,3] for T in T_camera2target_set]
            R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                                R_camera2target, t_camera2target,  
                                                                cv2.CALIB_HAND_EYE_PARK)
            T_base2camera = np.eye(4)
            T_base2camera[:3,:3] = R_camera2base
            T_base2camera[:3,3] = np.squeeze(T_base2camera)
            return T_base2camera        
                
        case _:
            print("Invalid method. Aborting...")
            return None