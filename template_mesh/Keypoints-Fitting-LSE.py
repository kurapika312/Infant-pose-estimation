import bpy
import math
import pathlib
import json

import numpy as np
import cv2

from mathutils import Vector, Matrix, Quaternion

# Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
# with least-squares error.
# Returns (scale factor c, rotation matrix R, translation vector t) such that
#   Q = P*cR + t
# if they align perfectly, or such that
#   SUM over point i ( | P_i*cR + t - Q_i |^2 )
# is minimised if they don't align perfectly.
def umeyama(P: np.ndarray, Q: np.ndarray):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t

def getDepthImage(depth_image_path: pathlib.Path) -> tuple:
    # load depth image using OpenCV - can be replaced by any other library that loads image to numpy array
    depth_im = cv2.imread(f'{depth_image_path}', -1)
    # Get the depth values from the depth image using the indices
    raveled_depth = depth_im.ravel()
    max_depth = np.max(raveled_depth)
    return depth_im, max_depth


def getJointsMapping(joints_json_path: pathlib.Path) -> dict:
    f = open(joints_json_path)
    joints_json = json.load(f)
    f.close()
    return joints_json


def getCameraIntrinsics(camera_intrinsics_path: pathlib.Path) -> tuple:
    camera_intrinsics: np.ndarray = np.load(camera_intrinsics_path)
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2].astype(
        float), camera_intrinsics[1, 2].astype(float)

    return fx, fy, cx, cy


def getJSONKeyPoints(json_path: pathlib.Path) -> list:
    f = open(json_path)
    annotations = json.load(f)
    f.close()
    return annotations['annotations']


def keypointsXYZ(flat_keypoints: list, depth_image_path: pathlib.Path, camera_intrinsics: tuple) -> np.ndarray:
    fx, fy, cx, cy = camera_intrinsics
    depth_image, max_depth = getDepthImage(depth_image_path)

    xyz_keypoints = []
    plane_points = [Vector((0, 0, 1)), Vector((511, 0, 1)), Vector((511, 511, 1)), Vector((0, 511, 1))]
    for i in range(0, len(flat_keypoints), 3):
        x, y = int(flat_keypoints[i]), int(flat_keypoints[i + 1])
        z = depth_image[y, x] / max_depth
        x3D, y3D, z3D = ((x - cx) * z) / fx, ((y - cy) * z) / fy, z
        xyz_keypoints.append([x3D, y3D, z3D])
    
    for p_point in plane_points:
        x, y, z = p_point
        x3D, y3D, z3D = ((x - cx) * z) / fx, ((y - cy) * z) / fy, z
        print(x3D, y3D, z3D)

    return np.array(xyz_keypoints)

def doLSE(context: bpy.types.Context, mesh: bpy.types.Object, mapping: dict, mapping_positions: np.ndarray):
    vgroups = mesh.vertex_groups
    hook_positions = np.zeros((len(mapping.keys()), 3))
    target_positions = np.zeros((mapping_positions.shape[0], 3))
    for vgroup in vgroups:
        if(mapping.get(vgroup.name, None)):
            index: int = mapping[vgroup.name]['mapped']
            position: np.ndarray = mapping_positions[index]
            hook: bpy.types.Object = context.view_layer.objects[vgroup.name]
            hook_positions[index] = hook.location
            target_positions[index] = position
            
            # hook.location = Vector(position)
            # print(position, hook)
    c, R, T = umeyama(hook_positions, target_positions)
    mesh.rotation_mode = 'QUATERNION'
    rotation_quat = Matrix(R).to_quaternion()
    rotation_quat.w *= -1
    mesh.rotation_quaternion = -rotation_quat
    mesh.scale = (c, c, c)
    mesh.location = Vector(T)
    context.view_layer.update()    
    
    for vgroup in vgroups:
        if(mapping.get(vgroup.name, None)):
            index: int = mapping[vgroup.name]['mapped']
            position: np.ndarray = mapping_positions[index]
            local_position: Vector = mesh.matrix_world.copy().inverted()@Vector(position)
            hook: bpy.types.Object = context.view_layer.objects[vgroup.name]
            hook.location = local_position
    

JSON_FILE_NAME: str = 'test4160_keypoints.json'
JSON_PATH: pathlib.Path = pathlib.Path(
    bpy.path.abspath('//')).joinpath('tests', JSON_FILE_NAME)
CAMERA_INTRINSICS_PATH: pathlib.Path = pathlib.Path(
    bpy.path.abspath('//')).joinpath('tests', 'camera_intrinsics.npy')
DEPTH_IMAGE_PATH: pathlib.Path = pathlib.Path(
    bpy.path.abspath('//')).joinpath('tests', '4160-DEPTH.png')
JOINTS_MAPPING_PATH: pathlib.Path = pathlib.Path(
    bpy.path.abspath('//')).joinpath('blender-to-coco-mapping.json')
JOINTS_MAPPING_PATH_TO_BLENDER: pathlib.Path = pathlib.Path(
    bpy.path.abspath('//')).joinpath('coco-to-blender-mapping.json')

C = bpy.context
mesh = C.view_layer.objects['Human_clean']

keypoints_flat: list = getJSONKeyPoints(JSON_PATH)
camera_intrinsics: tuple = getCameraIntrinsics(CAMERA_INTRINSICS_PATH)
joints_mapping: dict = getJointsMapping(JOINTS_MAPPING_PATH)
joints_mapping_blender: dict = getJointsMapping(JOINTS_MAPPING_PATH_TO_BLENDER)

keypoints_xyz: np.ndarray = keypointsXYZ(
    keypoints_flat, DEPTH_IMAGE_PATH, camera_intrinsics)

doLSE(C, mesh, joints_mapping, keypoints_xyz)
bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(f'//{JSON_PATH.parent.name}/{JSON_PATH.stem}.blend'))