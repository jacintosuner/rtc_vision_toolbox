import numpy as np
import cv2

def test():
    '''
    Generate 
        (A) a random set of transformation matrices (robot_base to target) or T_base2target_set
        (B) a random transformation matrix (robot_base to camera) or T_base2camera
    and calculate
        (C) a random set of transformation matrices (camera to target) or T_camera2target_set
    
    We use (A) and (C) above to test various approaches to calculate (B), the transformation matrix from robot_base to camera
    
    Assumption:
        1. 3D space
        2. target is on gripper, and T_gripper2target is identity matrix
        
    Note: 
        T_a2b represents frame 'b' in frame 'a', or
        tranforms a point from frame 'b' to frame 'a'
    '''
    print("Hello from test.py")
    
    SAMPLE_SIZE = 3
    
    # generate T_base2target_set
    T_base2target_set = []
    for i in range(SAMPLE_SIZE):
        T = get_random_transformation_matrix()
        T_base2target_set.append(T)
    
    # generate T_base2camera
    T_base2camera = get_random_transformation_matrix()
    print('T_base2camera=\n',T_base2camera)
    
    # calculate T_camera2target_set
    T_camera2target_set = []
    for i in range(SAMPLE_SIZE):
        T = np.dot(np.linalg.inv(T_base2camera), T_base2target_set[i])
        T_camera2target_set.append(T)
        
    # calculate T_base2camera using T_base2target_set and T_camera2target_set
    T_base2camera_calculated = []
    
    # method 1: one shot calculation
    T_base2camera_calculated = np.dot(T_base2target_set[0], np.linalg.inv(T_camera2target_set[0]))
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 1): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 1)=\n',T_base2camera_calculated)
    
    # method 2: solve_rigid_transformation
    t_camera2target = np.array([T[:3,3] for T in T_camera2target_set])
    t_base2target = np.array([T[:3,3] for T in T_base2target_set])
    T_base2camera_calculated = solve_rigid_transformation(t_base2target, t_camera2target)
    avg_error, std_error = calculate_reprojection_error(T_camera2target_set, T_base2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 2): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 2)=\n',T_base2camera_calculated)
    
    # method 3: opencv library. 
    # Details: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    T_target2base_set = [np.linalg.inv(T) for T in T_base2target_set]
    R_target2base = [T[:3,:3] for T in T_target2base_set]
    t_target2base = [T[:3,3] for T in T_target2base_set]
    
    R_camera2target = [T[:3,:3] for T in T_camera2target_set]
    t_camera2target = [T[:3,3] for T in T_camera2target_set]
    
    # CALIB_HAND_EYE_TSAI 
    R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                        R_camera2target, t_camera2target,  
                                                        cv2.CALIB_HAND_EYE_TSAI)
    T_base2camera_calculated = np.eye(4)
    T_base2camera_calculated[:3,:3] = R_camera2base
    T_base2camera_calculated[:3,3] = np.squeeze(T_base2camera)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 3 TSAI): {avg_error}, std. error: {std_error}")    
    print('T_base2camera_calculated (method 3 CALIB_HAND_EYE_TSAI )=\n',T_base2camera_calculated)
    
    # CALIB_HAND_EYE_ANDREFF 
    R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                        R_camera2target, t_camera2target,  
                                                        cv2.CALIB_HAND_EYE_ANDREFF)
    T_base2camera_calculated = np.eye(4)
    T_base2camera_calculated[:3,:3] = R_camera2base
    T_base2camera_calculated[:3,3] = np.squeeze(T_base2camera)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 3 ANDREFF): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 3 CALIB_HAND_EYE_ANDREFF )=\n',T_base2camera_calculated)
    
    # CALIB_HAND_EYE_PARK  
    R_camera2base, T_base2camera = cv2.calibrateHandEye(R_target2base, t_target2base, 
                                                        R_camera2target, t_camera2target,  
                                                        cv2.CALIB_HAND_EYE_PARK )
    T_base2camera_calculated = np.eye(4)
    T_base2camera_calculated[:3,:3] = R_camera2base
    T_base2camera_calculated[:3,3] = np.squeeze(T_base2camera)
    avg_error, std_error = calculate_reprojection_error(T_base2target_set, T_camera2target_set, T_base2camera_calculated)
    print(f"\nAvg. reprojection error (method 3 PARK): {avg_error}, std. error: {std_error}")
    print('T_base2camera_calculated (method 3 CALIB_HAND_EYE_PARK  )=\n',T_base2camera_calculated)

def get_random_transformation_matrix():
    '''
    Generate a random transformation matrix
    '''
    T = np.eye(4)
    
    # rotation matrix
    mat = np.random.rand(3,3)
    q,r = np.linalg.qr(mat)
    if np.linalg.det(q) < 0:
        q[:,0] *= -1
    T[:3,:3] = q
    
    # translation vector
    coords = np.random.randn(3)
    T[:3,3] = (coords / np.linalg.norm(coords)) * np.random.rand(1) * 10
    
    return T  
    
def solve_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """

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

def calculate_reprojection_error(T_a2t_set, T_b2t_set, T_a2b):
    errors = []
    assert len(T_a2t_set) == len(T_b2t_set)
    
    # Calculate error using transformation projections
    for i in range(len(T_b2t_set)):
        T_a2t_calc = np.dot(T_a2b, T_b2t_set[i])
        error = np.linalg.norm(T_a2t_set[i] - T_a2t_calc)
        errors.append(error)
    
    # Calculate error using pose projections
    # T_a2b = np.linalg.inv(T_b2a)
    # for i in range(len(T_t2a_set)):
    #     pose_target_in_frame_b = T_t2b_set[i][:,3]
    #     pose_target_in_frame_a = np.dot(T_a2b, pose_target_in_frame_b)
    #     error = np.linalg.norm(T_t2a_set[i][:,3] - pose_target_in_frame_a)
    #     errors.append(error)
        
    # Compute average error
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    return avg_error, std_error

if __name__ == "__main__":
    test()