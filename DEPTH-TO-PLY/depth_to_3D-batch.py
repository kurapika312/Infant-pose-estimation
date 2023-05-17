# (c) 2018 Fraunhofer IOSB
# see mini-rgbd-license.txt for licensing information

from os.path import join, exists
from os import makedirs
import pathlib
import tqdm

import numpy as np
import scipy
import cv2
import sklearn.decomposition as skdecompose

import pymeshlab

# camera calibration used for generation of depth
fx = 588.67905803875317
fy = 590.25690113005601
cx = 322.22048191353628
cy = 237.46785983766890

DOWN_SAMPLE_SIZE: int = -1 # Use -1 for no downsampling
PLY_HEADER_WITH_NORMALS_COLORS = 'ply\nformat ascii 1.0\ncomment Created by labeled_geometry.py\nelement vertex {0}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nproperty uchar label\nend_header'
PLY_FORMAT_WITH_NORMALS_COLORS = '%f %f %f %f %f %f %d %d %d %d %d'

'''
    PATH WHERE THE RGB, RGBD (DEPTH), SEGMENTATION IMAGES CAN BE FOUND
'''
BASE_PATH = pathlib.Path('/home/ashok/Workspace/LearnMLNN/projects/Infant-pose-estimation/pipeline/assets/Mannequin/').resolve()#pathlib.Path(__file__).parent.resolve()
DEPTH_IMAGES = BASE_PATH.joinpath('DEPTH').resolve()
OUTPUT_FOLDER = BASE_PATH.joinpath('PLY').resolve()
CAMERA_INTRINSICS = np.load(DEPTH_IMAGES.joinpath('camera_intrinsics.npy').resolve())


# BACKGROUND - 0
# FACE - 1
# ARMS - 2
# LEGS - 3
# THORAX - 4
VIDYA_REDUCED_SEGMENTATION_INDICES = np.array(
        [0, 2, 1, 3, 1, 1, 1, 1, 3, 4, 1, 3, 2, 2, 1, 1, 4, 1, 3, 1, 2]
    , dtype=int)
VIDYA_REDUCED_SEGMENTATION_COLORS = np.array(
    [
        [0, 0, 0, 255],
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 119, 0, 255],
    ], dtype=int)
VIDYA_SEGMENTATION_COLORS = np.array(
        [
            [0,	0,	0, 255],#	Background 0 
            [0,	0,	204, 255],#	Left Hand  2
            [0,	0,	255, 255],#	Right Eye 1
            [0,	102,	0, 255],#	Right Leg 3 
            [0,	255,	0, 255],#	Left Eye 1
            [0,	255,	255, 255],#	Left Cheek 1
            [128,	128,	128, 255],#	RestOfFace 1
            [135,	206,	250, 255],#	Right Ear 1
            [139,	0,	0, 255],#	Left Leg 3 
            [139,	69,	19, 255],#	Chest 4
            [144,	238,	144, 255],#	Left Ear 1
            [192,	192,	192, 255],#	Left Feet 3
            [240,	230,	140, 255],#	RightArm 2
            [245,	245,	220, 255],#	Right Hand 2
            [255,	0,	0, 255],#	Forehead 1
            [255,	0,	255, 255],#	Right Cheek 1
            [255,	153,	0, 255],#	Abdomen 4
            [255,	204,	153, 255],#	Lips 1
            [255,	215,	0, 255],#	Right Feet 3
            [255,	255,	0, 255],#	Nose 1
            [255,	255,	240, 255],#	Left Arm 2
        ], dtype=int)

fx, fy = CAMERA_INTRINSICS[0, 0], CAMERA_INTRINSICS[1, 1]
cx, cy = CAMERA_INTRINSICS[0, 2].astype(float), CAMERA_INTRINSICS[1, 2].astype(float)


# adjust path if necessary
# folder = BASE_PATH.joinpath('../01/depth')
# filename = folder.joinpath('syn_00000_depth.png')
# output_folder = BASE_PATH.joinpath('/home/ashok/Workspace/LearnMLNN/projects/PointStack/data/syntheticpartnormal/smil-PLY/')


def get2DFrom3D(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float)->np.ndarray:
    x_coords = np.floor(((points_3d[:, 0] * fx) / (points_3d[:, 2])) + cx).astype(int)
    y_coords = np.floor(((points_3d[:, 1] * fy) / (points_3d[:, 2])) + cy).astype(int)

    x_coords[x_coords < 0] = 0
    y_coords[y_coords < 0] = 0

    return np.vstack((y_coords, x_coords))

def savePLY(ply_np_data: np.ndarray, output_path: pathlib.Path, ply_headers_generic: str, ply_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ply_np_data = ply_np_data[~np.isnan(ply_np_data).any(axis=1), :]
    if(output_path.suffix == '.npy'):
        np.save(f'{output_path}', ply_np_data)
    else:
        ply_header = ply_headers_generic.format(ply_np_data.shape[0])    
        np.savetxt(f'{output_path}', ply_np_data, header=ply_header, comments='', encoding='ASCII', fmt=ply_format)

def project3D(rgb_image_path: pathlib.Path, 
              depth_image_path: pathlib.Path, 
              output_folder: pathlib.Path, 
              fx: float, fy: float, cx: float, cy: float, *, 
              segmentation_image: pathlib.Path=None, 
              down_sample_size: int = -1, 
              segmentation_colors: np.ndarray=np.zeros((0, 0, 4 )),
              reduced_segmentation_indices: np.ndarray = np.zeros((0)),
              reduced_segmentation_colors: np.ndarray = np.zeros((0, 0, 4)),
              ):
    
    #If a segmentation palette is not given to choose the label index from
    #Then just give 0 for black and 1 for white
    if(not segmentation_colors.shape[0]): 
        segmentation_colors = np.array([[0 ,0, 0, 255], [1, 1, 1, 255]])

    # Create the kd tree for searching through the segmentation colors
    segmentation_palette = scipy.spatial.cKDTree(segmentation_colors)

    rgb_im = cv2.imread(f'{rgb_image_path}', cv2.IMREAD_COLOR)
    rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGBA)

    # load depth image using OpenCV - can be replaced by any other library that loads image to numpy array
    depth_im = cv2.imread(f'{depth_image_path}', -1)

    im_width, im_height = depth_im.shape
    
    # create tuple containing image indices
    indices = tuple(np.mgrid[:depth_im.shape[0],:depth_im.shape[1]].reshape((2,-1)))   

    # Get the depth values from the depth image using the indices
    raveled_depth = depth_im[indices].ravel()
    raveled_seg_rgb: np.ndarray = np.zeros((0, 0, 3))
    
    #Create an empty (im_width * im_height) x 3 array for the 3d points
    pts3D = np.zeros((indices[0].size, 3))
    #The z value is depth at a pixel / the distance of that pixel from kinect camera
    pts3D[:, 2] = raveled_depth / np.max(raveled_depth) #1000.
    
    # Get the 3D x coordinate from the 2D pixel x by offseting 
    # the center x (cx) from the pixel 
    # Multiply this by the depth value of that pixel
    # Finally divide the whole thing by field of view X (fx)
    pts3D[:, 0] = ((np.asarray(indices).T[:, 1] - cx) * pts3D[:, 2]) / fx
    # Similar to the 3D X coordinate calculate the 3D Y coordinate 
    pts3D[:, 1] = ((np.asarray(indices).T[:, 0] - cy) * pts3D[:, 2]) / fy

    # write to .obj file
    output_folder.mkdir(exist_ok=True, parents=True)

    #Given a segmentation or mask image choose only non-black pixels
    if(segmentation_image):
        seg_rgb_im = cv2.imread(f'{segmentation_image}', cv2.IMREAD_COLOR)
        seg_rgb_im = cv2.cvtColor(seg_rgb_im, cv2.COLOR_BGR2RGBA)
        raveled_seg_rgb = seg_rgb_im.reshape(im_height*im_width, 4)

        mask_region_indices = ~np.all(raveled_seg_rgb == [0, 0, 0, 255], axis=-1)
        pts3D = pts3D[mask_region_indices]

    ms: pymeshlab.MeshSet = pymeshlab.MeshSet()
    m: pymeshlab.Mesh = pymeshlab.Mesh(pts3D)
    ms.add_mesh(m, depth_image_path.stem)
    ms.compute_normal_for_point_clouds(k=1000, flipflag=True)   

    if(down_sample_size > -1):
        # point cloud sampling here to restrict for 2048 or 4096 or as much as you need
        ms.generate_simplified_point_cloud(samplenum=down_sample_size)
        
    m = ms.current_mesh()

    #The 3d points in the point cloud
    v: np.ndarray = m.vertex_matrix()
    #The 3D normals in the point cloud
    n: np.ndarray = m.vertex_normal_matrix()

    #Based on the final 3D positions (after segmentation, downsampling) get their respective 2d coordinates again
    coords_2d: np.ndarray = get2DFrom3D(v, fx, fy, cx, cy)
    #Based on the 2d coordinates find their respective colors in the segmentation or RGB image again
    c: np.ndarray = rgb_im[coords_2d[0], coords_2d[1]]    # Use the colors from real texture
    

    #Time to fill the labels of each point in the point cloud
    #This is done by going through the colors of each point and assign the respective class index
    l: np.ndarray = np.ones((v.shape[0], 1))#np.ones((raveled_rgb.shape[0], 1))    

    if(segmentation_image):
        segmentation_c: np.ndarray = seg_rgb_im[coords_2d[0], coords_2d[1]]  # Use the colors from segmentation texture
        # Find the indices to use as label index for each point with the segmentation color
        _, indices = segmentation_palette.query(segmentation_c)

    if(reduced_segmentation_indices.shape[0]):        
        indices = reduced_segmentation_indices[indices]
        '''
            Enable the below line for debugging purposes to 
            show the segemntati0on colors as opposed to rendered 
            texture colors
        '''
        # c = reduced_segmentation_colors[indices]
    
    indices = indices.reshape(indices.shape[0], 1)

    l[:,] = indices[:,]
    
    ply_np_data = np.hstack((v, n, c, l))

    return ply_np_data    

def normalize_pc(points)->np.ndarray:
    centroid = np.mean(points, axis = 0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    points /= furthest_distance
    return points

def pcaAlignedPointCloud(ply_np_data: np.ndarray, normalize: bool = True)->np.ndarray:

    ms: pymeshlab.MeshSet = pymeshlab.MeshSet()
    m: pymeshlab.Mesh = None
    normals: np.ndarray

    pca: skdecompose.PCA = skdecompose.PCA(n_components=3)
    pca_points: np.ndarray = ply_np_data[:,:3]
    pca_ply_np_data: np.ndarray = pca.fit_transform(pca_points)

    if(normalize):
        pca_ply_np_data = normalize_pc(pca_ply_np_data)
    

    min_pt = np.min(pca_ply_np_data, axis=0)
    max_pt = np.max(pca_ply_np_data, axis=0)
    size = max_pt - min_pt
    center = min_pt + (size * 0.5)
    view_dir = np.zeros((3))
    view_dir[np.argmin(size)] = -10000.0
    view_dir[np.argmax(size)] = -10000.0
    view_pos = center + view_dir
    # print(min_pt, max_pt, size, np.argmax(size))
    # print(view_dir)
    
    m = pymeshlab.Mesh(pca_ply_np_data)
    ms.add_mesh(m, 'pca_mesh')
    ms.compute_normal_for_point_clouds(k=50, flipflag=True, viewpos=view_pos)        
    
    m = ms.current_mesh()
    normals = m.vertex_normal_matrix()
    pca_ply_np_data = np.hstack((pca_ply_np_data, normals, ply_np_data[:,6:]))
    return pca_ply_np_data

def downsampled(indices: np.ndarray, unique_indices: list = [0, 1, 2, 3, 4], total_points: int = 3000)->np.ndarray:
    downsampled_indices = np.zeros((0))
    npoints = int(float(total_points) / float(len(unique_indices)))
    for uid in unique_indices:
        current_indices = np.argwhere(indices == uid)
        current_indices = current_indices.flatten()
        min_n_points = min(current_indices.shape[0], npoints)
        choice = np.random.choice(current_indices, min_n_points, replace=False)
        downsampled_indices = np.hstack((downsampled_indices, choice))

    return downsampled_indices.astype(int)

def batchDepthToPLY(
        depth_folder: pathlib.Path, 
        output_folder: pathlib.Path, 
        fx: float, fy: float, cx: float, cy: float, *,
        save_pca_cloud: bool = False):
    

    glob_result: pathlib.Path.glob = depth_folder.glob('*.png')
    output_folder.mkdir(parents=True, exist_ok=True)
    glob_result = list(glob_result)

    for i, d_png in tqdm.tqdm(enumerate(glob_result), dynamic_ncols=True, total=len(glob_result)):
        # print(d_png.stem)
        rgb_name: str = d_png.stem.split('-')[0]
        rgb_image: pathlib.Path = d_png.parent.parent.joinpath('RGB').joinpath(f'{rgb_name}-RGB.png')
        seg_rgb_image: pathlib.Path = d_png.parent.parent.joinpath('SEGMENTATION').joinpath(f'{rgb_name}-SEGMENTATION.png')
        # print(rgb_name)
        # raise IndexError
        ply_np_data: np.ndarray = project3D(
                  rgb_image, d_png, 
                  output_folder, 
                  fx, fy, cx, cy, 
                  down_sample_size=DOWN_SAMPLE_SIZE, 
                  segmentation_image=seg_rgb_image,
                  segmentation_colors=VIDYA_SEGMENTATION_COLORS, 
                  reduced_segmentation_indices=VIDYA_REDUCED_SEGMENTATION_INDICES, 
                  reduced_segmentation_colors=VIDYA_REDUCED_SEGMENTATION_COLORS)
        

        downsampled_indices = downsampled(ply_np_data[:,-1], [1, 2, 3, 4], total_points=3000)
        ply_np_data = ply_np_data[downsampled_indices]


        output_path = output_folder.joinpath(f'{d_png.stem}.npy')
        savePLY(ply_np_data, output_path, PLY_HEADER_WITH_NORMALS_COLORS, PLY_FORMAT_WITH_NORMALS_COLORS)

        if(save_pca_cloud):
            output_pca_folder = output_folder.parent.joinpath('PCA_PLY')
            output_pca_folder.mkdir(parents=True, exist_ok=True)
            output_path_pca = output_pca_folder.joinpath(f'{d_png.stem}.npy')
            pca_ply_np_data = pcaAlignedPointCloud(ply_np_data)
            savePLY(pca_ply_np_data, output_path_pca, PLY_HEADER_WITH_NORMALS_COLORS, PLY_FORMAT_WITH_NORMALS_COLORS)

        # if(i >= 0):
        #     break

if __name__ == '__main__':
    batchDepthToPLY(DEPTH_IMAGES, OUTPUT_FOLDER, fx, fy, cx, cy, save_pca_cloud=True)