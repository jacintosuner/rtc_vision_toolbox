import open3d as o3d

from camera.orbbec.ob_camera import OBCamera

def main():
    camera_index = 0
    camera = OBCamera(device_index=camera_index)

    while True:
        try:
            pcd = camera.get_point_cloud(
                min_mm = 400,
                max_mm = 1000,
                save_points = False,
                use_new_frame = True
            )
            if pcd is not None:
                o3d.visualization.draw_geometries([pcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
        except Exception as e:
            raise e

if __name__ == "__main__":
    main()
