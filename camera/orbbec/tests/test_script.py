from camera.orbbec.ob_camera import OBCamera
import cv2
import numpy as np

cam_sn = {
    'cam0': 'CL8FC3100RL',
    'cam1': 'CL8FC3100W3',
    'cam2': 'CL8FC3100NM',
}

# TEST1: GET RGB/DEPTH INTRINSICS. WORKS OK ✓
def test1(camera):        
    rgb_intrinsics = camera.get_rgb_intrinsics()
    depth_intrinsics = camera.get_depth_intrinsics()
    print("RGB intrinsics: \n", rgb_intrinsics)
    print("Depth intrinsics: \n", depth_intrinsics)
    
# TEST2: GET RGB IMAGE. WORKS OK ✓
def test2(camera):
    image = camera.get_rgb_image()
    cv2.imwrite('rgb_image.png', image)

# TEST3: GET DEFAULT DEPTH IMAGE and DATA. WORKS OK ✓
def test3(camera):
    depth_image = camera.get_raw_depth_data()    
    
    print("Depth image shape: ", depth_image.shape)
    print("Depth image type: ", depth_image.dtype)
    print("Depth image: \n", depth_image[:4, :4])
    print("Max depth value: ", np.nanmax(depth_image))
    print("Min depth value: ", np.nanmin(depth_image))
    
    np.save('depth_data.npy', depth_image)

    depth_image = camera.get_depth_image()
    
    cv2.imwrite('depth_image.png', depth_image)
    
    depth_image = camera.get_colormap_depth_image()
    cv2.imwrite('depth_image_colormap.png', depth_image)
    
# TEST4: GET DEFAULT POINT CLOUD.
def test4(camera):
    print("Getting default point cloud")
    point_cloud = camera.get_point_cloud(save_points=True)
    print("Point cloud shape: ", np.asarray(point_cloud.points).shape)
    
if __name__ == "__main__":
    
    print("Testing OBCamera class")
    
    cam_id = input("Enter camera id (0/1/2): ")
    camera = OBCamera(serial_no=cam_sn[f'cam{cam_id}'])
    
    print("Available tests: ")
    print("1. Get RGB/DEPTH intrinsics")
    print("2. Get RGB image")
    print("3. Get depth image")
    print("4. Get point cloud")
    test = input("Enter test number (1-6): ")
    
    match test:
        case '1':
            test1(camera)
        case '2':
            test2(camera)
        case '3':
            test3(camera)
        case '4':
            test4(camera)
        case _:
            print("Invalid test number")
