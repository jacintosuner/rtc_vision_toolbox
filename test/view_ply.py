import glob
import os
import re

import cv2
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.graph_objs as go
from scipy.spatial.transform import Rotation as R

from camera.zed_ros.zed_ros import ZedRos

def plot_multi_np(plist):
    """
    Plots multiple point clouds in the same plot using plotly
    Assumes points are in meters coordinates.
    
    Args: 
        plist: list of np arrays of shape (N, 3)
    """
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
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
    
    colors = ['red', 'green', 'blue']  # X, Y, Z axis colors
    
    fig = go.Figure()
    
    T = []
    T.append(np.eye(4))

    for tf in T:
        origin = tf[:3, 3]
        axes = tf[:3, :3]

        for i in range(3):
            axis_end = origin + 0.1*axes[:, i]
            fig.add_trace(go.Scatter3d(
                x=[origin[0], axis_end[0]],
                y=[origin[1], axis_end[1]],
                z=[origin[2], axis_end[2]],
                mode='lines',
                line=dict(color=colors[i], width=4),
                name='Axis ' + str(i+1)
            ))

    for plot in go_data:
        fig.add_trace(plot)
    
    # add axis lines and camera view
    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        camera = dict(
                      eye=dict(x=-1.30, y=0, z=-.25),
                      center=dict(x=0., y=0, z=-0.25),
                      up=dict(x=0, y=0, z=1),
                     )
        ),
        height=800,
        width=1200,
    )    
    
    fig.show()
    
# pcd = o3d.io.read_point_cloud('/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/09-11-usb/teach_data/pcd_data/demo0_ih_camera_view0_cam3_gripper_pointcloud.ply')
# pcd = o3d.io.read_point_cloud('/home/mfi/repos/rtc_vision_toolbox/camera/rs_ros/points_came_igev_0911_1630.ply')
# pcd = o3d.io.read_point_cloud('/home/mfi/repos/rtc_vision_toolbox/data/demonstrations/09-08-dsub/teach_data/pcd_data/demo0_gripper_close_up_view0_cam2_closeup_pointcloud.ply')
# pcd = pcd.uniform_down_sample(35)
# points = np.asarray(pcd.points)

points = np.load("/home/mfi/repos/rtc_vision_toolbox/test/demo0_anchor_og.npy")
# points = points[np.where(points[:, 2] < 200)]
# points[:,2] = -points[:,2]
plot_multi_np([points])