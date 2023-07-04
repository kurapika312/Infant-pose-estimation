import pathlib
import numpy as np

import plyfile


from depth_to_3D_batch import getCameraIntrinsics, get2DFrom3D

image_size: tuple = (1080, 1920) # Height x Width
camera_intrinsics_file: pathlib.Path = pathlib.Path('azure_camera_intrinsics.npy')
ply_files_directory: pathlib.Path = pathlib.Path('/home/ashok/Workspace/Segmentation-3D/Research Project-20230125T175425Z-001/Research Project/2-gold_standard_35_az1/')


def readPLY(ply_path: pathlib.Path)->np.ndarray:
    plydata: plyfile.PlyData = plyfile.PlyData.read(f'{ply_path.resolve()}')
    vertices: np.array = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    colors:np.array = np.vstack((plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue'])).T    
    return vertices, colors

def getPlyFiles(base_directory: pathlib.Path)->list[pathlib.Path]:
    glob_result = [f for f in base_directory.glob('*.ply') if f.parent == base_directory]
    return sorted(glob_result)

def getRGBImage(fx: float, fy: float, cx: float, cy: float, image_size: tuple, ply_data_vertices: np.ndarray, ply_data_colors: np.ndarray)->np.ndarray:
    np_rgb_image: np.ndarray = np.zeros((image_size[0], image_size[1], 4))
    np_depth_image: np.ndarray = np.zeros((image_size[0], image_size[1]))
    np_image_coordinates: np.ndarray = get2DFrom3D(ply_data_vertices, fx, fy, cx, cy)
    y_coords, x_coords = np_image_coordinates
    print(np.max(y_coords), np.max(x_coords))
    np_rgb_image[y_coords, x_coords] = 135    


if __name__ == '__main__':
    fx, fy, cx, cy = getCameraIntrinsics(camera_intrinsics_file)
    ply_files_result: list[pathlib.Path] = getPlyFiles(ply_files_directory)
    for ply_file in ply_files_result:
        vertices, colors = readPLY(ply_file)
        getRGBImage(fx, fy, cx, cy, image_size, vertices, colors)
        break