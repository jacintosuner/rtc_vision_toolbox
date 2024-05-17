from camera.zed_ros.zed_ros import ZedRos
from camera.orbbec.ob_camera import OBCamera
import cv2
import numpy as np
import glob
import re
import open3d as o3d
import plotly.graph_objs as go

def plot_multi_np(plist):
    """
    Args: plist, list of numpy arrays of shape, (1,num_points,3)
    """
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#8c564b',  # chestnut brown
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#8c564b',  # chestnut brown
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#8c564b',  # chestnut brown
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
    ]
    skip = 1
    go_data = []
    for i in range(len(plist)):
        p_dp = plist[i]
        plot = go.Scatter3d(x=p_dp[::skip,0], y=p_dp[::skip,1], z=p_dp[::skip,2], 
                     mode='markers', marker=dict(size=2, color=colors[i],
                     symbol='circle'))
        go_data.append(plot)
 
    layout = go.Layout(
        scene=dict(
            aspectmode='data',
        ),
        height=1200,
        width=1200,
    )

    fig = go.Figure(data=go_data, layout=layout)
    fig.show()
    return fig

cam0 = ZedRos(camera_node=f'/cam0/zed_cam0', camera_type='zedx', depth_mode='igev')
# cam1 = ZedRos(camera_node=f'/cam1/zed_cam1', camera_type='zedx', depth_mode='igev')
# cam2 = ZedRos(camera_node=f'/cam2/zed_cam2', camera_type='zedxm', depth_mode='igev')
ob_cam0 = OBCamera(serial_no='CL8FC3100RL')

# cameras = [cam0, cam1, cam2]
cameras = [cam0]

pcd = np.asarray(cam0.get_point_cloud().uniform_down_sample(25).points)
print(pcd.shape)

ob_pcd = np.asarray(ob_cam0.get_point_cloud().uniform_down_sample(25).points)
print(ob_pcd.shape)

plot_multi_np([pcd, ob_pcd])