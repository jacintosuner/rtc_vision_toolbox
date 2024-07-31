import argparse
import os

import numpy as np

from calibration.calibrations import *

"""
T_eef2marker = np.array([
                [-1.0, 0.0, 0.0, 0.1016], 
                [0.0, -1.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, 0.049], 
                [0.0, 0.0, 0.0, 1.0]])
"""

T_eef2marker = np.array(
    [
        [0.0, 0.0, 1.0, 0.041502],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.080824],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

print_tf = False


def save_tf(method, data_dir, save_path, sample_size=0):
    T_camera2marker_set = np.load(os.path.join(data_dir, "T_camera2marker_set.npy"))
    T_base2eef_set = np.load(os.path.join(data_dir, "T_base2eef_set.npy"))

    T_base2marker_set = [
        np.dot(T_base2eef, T_eef2marker) for T_base2eef in T_base2eef_set
    ]

    # create a TRAINING SET by choosing random distinct samples
    if sample_size == 0:
        sample_size = len(T_base2marker_set)
    random_integer_set = np.random.choice(
        len(T_base2marker_set), sample_size, replace=False
    )
    T_base2marker_training = [T_base2marker_set[i] for i in random_integer_set]
    T_camera2marker_training = [T_camera2marker_set[i] for i in random_integer_set]

    print("SAMPLE_SIZE:", sample_size)
    print("random_integer_set:", np.sort(random_integer_set))

    print(f"\nMETHOD: {method}")
    T_base2camera = solve_rigid_transformation(
        T_base2marker_training, T_camera2marker_training, method=method
    )
    avg_error, std_error = calculate_reprojection_error(
        T_base2marker_set, T_camera2marker_set, T_base2camera
    )
    print(f"Transformation matrix T_base2camera:\n{np.round(T_base2camera,2)}")
    print(f"Avg. reprojection error:: {avg_error}, std. error: {std_error}")

    np.save(save_path, T_base2camera)


def test(data_dir, sample_size=0):
    T_camera2marker_set = np.load(os.path.join(data_dir, "T_camera2marker_set.npy"))
    T_base2eef_set = np.load(os.path.join(data_dir, "T_base2eef_set.npy"))

    T_base2marker_set = [
        np.dot(T_base2eef, T_eef2marker) for T_base2eef in T_base2eef_set
    ]

    print("size of T_camera2marker_set:", len(T_camera2marker_set))
    print("size of T_base2marker_set:", len(T_base2marker_set))

    # create a TRAINING SET by choosing random distinct samples
    if sample_size == 0:
        sample_size = len(T_base2marker_set)
    random_integer_set = np.random.choice(
        len(T_base2marker_set), sample_size, replace=False
    )
    T_base2marker_training = [T_base2marker_set[i] for i in random_integer_set]
    T_camera2marker_training = [T_camera2marker_set[i] for i in random_integer_set]

    print("SAMPLE_SIZE:", sample_size)
    print("random_integer_set:", np.sort(random_integer_set))

    # T_base2marker_training = T_base2marker_set
    # T_camera2marker_training = T_camera2marker_set

    # print('size of T_base2marker_training:', len(T_base2marker_training))
    # print('size of T_camera2marker_training:', len(T_camera2marker_training))

    # print('T_base2eef=\n', T_base2eef_set[0])
    # print('T_base2marker=\n', T_base2marker_set[0])
    # print('T_eef2marker=\n', T_eef2marker)
    # print('T_camera2marker=\n', T_camera2marker_set[2])

    print("\nMETHOD1: ONE_SAMPLE_ESTIMATE")
    T_base2camera = solve_rigid_transformation(
        T_base2marker_training, T_camera2marker_training, method="ONE_SAMPLE_ESTIMATE"
    )
    avg_error, std_error = calculate_reprojection_error(
        T_base2marker_set, T_camera2marker_set, T_base2camera
    )
    if print_tf:
        print(f"Transformation matrix T_base2camera:\n{np.round(T_base2camera,2)}")
    print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")

    print("\nMETHOD2: SVD_ALGEBRAIC")
    T_base2camera = solve_rigid_transformation(
        T_base2marker_training, T_camera2marker_training, method="SVD_ALGEBRAIC"
    )
    avg_error, std_error = calculate_reprojection_error(
        T_base2marker_set, T_camera2marker_set, T_base2camera
    )
    if print_tf:
        print(f"Transformation matrix T_base2camera:\n{np.round(T_base2camera,2)}")
    print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")

    print("\nMETHOD3: CALIB_HAND_EYE_TSAI")
    T_base2camera = solve_rigid_transformation(
        T_base2marker_training, T_camera2marker_training, method="CALIB_HAND_EYE_TSAI"
    )
    avg_error, std_error = calculate_reprojection_error(
        T_base2marker_set, T_camera2marker_set, T_base2camera
    )
    if print_tf:
        print(f"Transformation matrix T_base2camera:\n{np.round(T_base2camera,2)}")
    print(f"Avg. reprojection error: {avg_error}, std. error: {std_error}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Save transformation matrix from calibration data"
    )

    parser.add_argument("--cam_type", type=str, help="Camera type (1-femto or 2-zed)")
    parser.add_argument("--cam_id", type=str, help="Use 0 or 1 or 2")
    parser.add_argument("--date", type=str, help="Date of calibration data (mm-dd)")
    parser.add_argument(
        "--method",
        type=str,
        help="Method to use for calibration (ONE_SAMPLE_ESTIMATE, SVD_ALGEBRAIC, CALIB_HAND_EYE_TSAI)",
    )
    parser.add_argument(
        "--tf_print", action="store_true", help="Print transformation matrix"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print_tf = args.tf_print

    cam_type = ""
    if args.cam_type is None:
        cam_type = input("Enter camera type (1-femto or 2-zed): ")
    else:
        cam_type = args.cam_type
    if cam_type == "1":
        cam_type = "femto"
    elif cam_type == "2":
        cam_type = "zed"

    cam_id = ""
    if args.cam_id is None:
        cam_id = input("Enter camera id (0, 1, 2): ")
    else:
        cam_id = args.cam_id

    date = ""
    if args.date is None:
        date = input("Enter date (e.g. 05-03): ")
    else:
        date = args.date

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(
        dir_path, f"../data/calibration_data/{cam_type}/cam{cam_id}/{date}"
    )
    save_path = os.path.join(
        dir_path, f"../data/calibration_data/{cam_type}/T_base2cam{cam_id}.npy"
    )

    print("------------------------------")
    print("Evaluating calibrations:")
    sample_size = input("Enter sample size (0 for all): ")
    sample_size = int(sample_size)
    test(data_dir, sample_size = sample_size)
    print("------------------------------")

    method = ""
    if args.method is None:
        method = input("Choose method for calibration (1/2/3):")
        match method:
            case "1":
                method = "ONE_SAMPLE_ESTIMATE"
            case "2":
                method = "SVD_ALGEBRAIC"
            case "3":
                method = "CALIB_HAND_EYE_TSAI"
            case _:
                method = "SVD_ALGEBRAIC"
    else:
        method = args.method

    save_data: bool = input(f"Save calibration using {method} method? (y/n): ")

    if save_data == "y":
        save_tf(method, data_dir, save_path)
        print("Calibration data saved. Bye.")
    else:
        print("Calibration data not saved. Bye.")
