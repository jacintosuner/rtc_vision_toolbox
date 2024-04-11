from calibration.calibrations import *
import numpy as np
import os
import argparse

T_eef2marker = np.array([
                [-1.0, 0.0, 0.0, 0.1016], 
                [0.0, -1.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, 0.049], 
                [0.0, 0.0, 0.0, 1.0]])

def save_tf(method, read_path, save_path):
    T_camera2marker_set = np.load(os.path.join(data_dir, 'T_camera2marker_set.npy'))
    T_base2eef_set = np.load(os.path.join(data_dir, 'T_base2eef_set.npy'))
    
    T_base2marker_set = [np.dot(T_base2eef, T_eef2marker) for T_base2eef in T_base2eef_set]
    
    print(f"\nMETHOD: {method}")
    T_base2camera = solve_rigid_transformation(T_base2marker_set, T_camera2marker_set, method=method)
    avg_error, std_error = calculate_reprojection_error(T_base2marker_set, T_camera2marker_set, T_base2camera)
    print(f"Transformation matrix T_base2camera:\n{T_base2camera}")
    print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")
    
    np.save(save_path, T_base2camera)        

def test():
    SAMPLE_SIZE = 10
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, '../../data/calibration_data/cam2')
    
    print(data_dir)
    
    T_camera2marker_set = np.load(os.path.join(data_dir, 'T_camera2marker_set.npy'))
    T_base2eef_set = np.load(os.path.join(data_dir, 'T_base2eef_set.npy'))
    
    T_base2marker_set = [np.dot(T_base2eef, T_eef2marker) for T_base2eef in T_base2eef_set]
    
    # create a TRAINING SET by choosing random distinct samples
    random_integer_set = np.random.choice(len(T_base2marker_set), SAMPLE_SIZE, replace=False)
    T_base2marker_training = [T_base2marker_set[i] for i in random_integer_set]
    T_camera2marker_training = [T_camera2marker_set[i] for i in random_integer_set] 
    
    print('SAMPLE_SIZE:', SAMPLE_SIZE)
    print('random_integer_set:', random_integer_set)   
    
    # print('T_base2eef=\n', T_base2eef_set[0])
    # print('T_base2marker=\n', T_base2marker_set[0])
    print('T_camera2marker=\n', T_camera2marker_set[2])
    
    # print('size of T_camera2marker_set:', len(T_camera2marker_set))
    # print('size of T_base2marker_set:', len(T_base2marker_set))
    
    print("\nMETHOD1: ONE_SAMPLE_ESTIMATE")
    T_base2camera = solve_rigid_transformation(T_base2marker_set, T_camera2marker_set, method="ONE_SAMPLE_ESTIMATE")
    avg_error, std_error = calculate_reprojection_error(T_base2marker_set, T_camera2marker_set, T_base2camera)
    #print(f"Transformation matrix T_base2camera:\n{T_base2camera}")
    print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")

    print("\nMETHOD2: SVD_ALGEBRAIC")
    T_base2camera = solve_rigid_transformation(T_base2marker_training, T_camera2marker_training, method="SVD_ALGEBRAIC")
    avg_error, std_error = calculate_reprojection_error(T_base2marker_set, T_camera2marker_set, T_base2camera)
    #print(f"Transformation matrix T_base2camera:\n{T_base2camera}")
    print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")
    
    print("\nMETHOD3: CALIB_HAND_EYE_TSAI")
    T_base2camera = solve_rigid_transformation(T_base2marker_training, T_camera2marker_training, method="CALIB_HAND_EYE_TSAI")
    avg_error, std_error = calculate_reprojection_error(T_base2marker_set, T_camera2marker_set, T_base2camera)
    #print(f"Transformation matrix T_base2camera:\n{T_base2camera}")
    print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")
    
def get_args():
    parser = argparse.ArgumentParser(description="Save transformation matrix from calibration data")
    
    parser.add_argument("--cam_id", type=str, help="use cam0 or cam1 or cam2 data")
    parser.add_argument("--method", type=str, help="method to use for calibration", default="ONE_SAMPLE_ESTIMATE")
    
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    #test()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, '../../data/calibration_data/'+args.cam_id)
    save_path = os.path.join(dir_path, f'../../data/calibration_data/T_base2{args.cam_id}.npy')
    save_tf(args.method, data_dir, save_path)
