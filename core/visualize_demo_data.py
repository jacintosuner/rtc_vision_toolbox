import numpy as np
import open3d as o3d     
import glob

log_root_dir = '/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/place_demo_pcds/2024-04-05-17-48-20'
files = glob.glob(f'{log_root_dir}/*.npz')
print(files)

for file in files:
    data = np.load(file, allow_pickle=True)

    filename = file.split('/')[-1]
    print(f'Showing {filename}')

    pcd = data['arr_0']

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)

    o3d.visualization.draw_geometries([o3d_pcd])
