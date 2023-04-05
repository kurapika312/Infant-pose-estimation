import bpy
import pathlib
import json

import numpy as np
import cv2

def getDepthImage(depth_image_path: pathlib.Path) -> tuple:
    # load depth image using OpenCV - can be replaced by any other library that loads image to numpy array
    depth_im = cv2.imread(f'{depth_image_path}', -1)
    # Get the depth values from the depth image using the indices
    raveled_depth = depth_im.ravel()
    max_depth = np.max(raveled_depth)
    return depth_im, max_depth

def getJointsMapping(joints_json_path: pathlib.Path)->list:
    f = open(joints_json_path)
    joints_json = json.load(f)
    f.close()
    return joints_json

def getCameraIntrinsics(camera_intrinsics_path: pathlib.Path) -> tuple:
    camera_intrinsics: np.ndarray = np.load(camera_intrinsics_path)
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2].astype(float), camera_intrinsics[1, 2].astype(float)

    return fx, fy, cx, cy

def getJSONKeyPoints(json_path: pathlib.Path)->np.ndarray:
    f = open(json_path)
    annotations = json.load(f)
    f.close()
    return annotations['annotations']

def keypointsXYZ(flat_keypoints: list, depth_image_path: pathlib.Path, camera_intrinsics: tuple)->np.ndarray:
    fx, fy, cx, cy = camera_intrinsics
    depth_image, max_depth = getDepthImage(depth_image_path)
    
    xyz_keypoints = []
    for i in range(0, len(flat_keypoints), 3):
        x, y = int(flat_keypoints[i]), int(flat_keypoints[i + 1])
        z = depth_image[y, x] / max_depth
        x3D, y3D, z3D = ((x - cx) * z) / fx, ((y - cy) * z) / fy, z
        xyz_keypoints.append([x3D, y3D, z3D])

    return np.array(xyz_keypoints)

def poseTemplate(armature_ob: bpy.types.Object, position: np.ndarray, mapping: dict)->None:
    print(len(armature_ob.edit_bones))
    for edit_bone in armature_ob.data.edit_bones:
        # edit_bone.location = (0, 0, 0)  
        print(edit_bone.location)

    

JSON_FILE_NAME: str = 'test4160_keypoints.json'
JSON_PATH: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('tests', JSON_FILE_NAME)
CAMERA_INTRINSICS_PATH: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('tests', 'camera_intrinsics.npy')
DEPTH_IMAGE_PATH: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('tests', '4160-DEPTH.png')
JOINTS_MAPPING_PATH: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('blender-to-coco-mapping.json')

C = bpy.context
armature = C.view_layer.objects['Coco17']

keypoints_flat = getJSONKeyPoints(JSON_PATH)
camera_intrinsics = getCameraIntrinsics(CAMERA_INTRINSICS_PATH)
joints_mapping = getJointsMapping(JOINTS_MAPPING_PATH)

keypoints_xyz = keypointsXYZ(keypoints_flat, DEPTH_IMAGE_PATH, camera_intrinsics)

poseTemplate(armature, keypoints_xyz, joints_mapping)