import bpy
import math
import pathlib
import json

import numpy as np
import cv2

from mathutils import Vector, Matrix, Quaternion


def getRotationScale(vectorA: Vector, vectorB: Vector) -> tuple:
    scale: float = vectorB.length / vectorA.length
    return vectorA.rotation_difference(vectorB), scale


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
    for i in range(0, len(flat_keypoints), 3):
        x, y = int(flat_keypoints[i]), int(flat_keypoints[i + 1])
        z = depth_image[y, x] / max_depth
        x3D, y3D, z3D = ((x - cx) * z) / fx, ((y - cy) * z) / fy, z
        xyz_keypoints.append([x3D, y3D, z3D])

    return np.array(xyz_keypoints)


def poseTemplateReset(armature_ob: bpy.types.Object) -> None:
    for pose_bone in armature_ob.pose.bones:
        pose_bone.rotation_quaternion = Quaternion((0, 0, 0), 1)
        pose_bone.scale = (1, 1, 1)
        pose_bone.location = (0, 0, 0)


def setDuplicateArmaturePosition(armature_object: bpy.types.Object, positions: np.ndarray, mapping_to_blender: dict):
    # print(armature_object.hide_get())
    # if (armature_object.hide_get()):
    #     return
    armature_object.hide_set(False)
    bpy.ops.object.select_all(action="DESELECT")
    armature_object.select_set(True)
    C.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode='EDIT')

    armature = armature_object.data
    
    for i, p in enumerate(positions):
        mapped_o = mapping_to_blender[str(i)]
        mapped_bone_name = mapped_o['mapped']
        edit_bone = armature.edit_bones[mapped_bone_name]
        pose_bone = armature_object.pose.bones[edit_bone.name]
        edit_bone.head = p
        
    for edit_bone in armature.edit_bones:
        if (not len(edit_bone.children)):
            edit_bone.tail = edit_bone.head + edit_bone.vector*0.01#Vector((0.0, 0, 0.002))
        else:
            np_child_heads = np.zeros((0, 3))
            for c_bone in edit_bone.children:
                np_child_heads = np.vstack(
                    (np_child_heads, np.array(c_bone.head)))

            edit_bone.tail = np.mean(np_child_heads, axis=0)

    bpy.ops.object.mode_set(mode='OBJECT')
    armature_object.hide_set(True)

def poseBonePosition(context: bpy.types.Context, boneName: str, from_armature: bpy.types.Object, to_armature: bpy.types.Object, mapping: dict, global_positions: np.ndarray) -> None:   
    
    global_position = Vector(global_positions[mapping[boneName]['mapped']])
    pose_bone = from_armature.pose.bones[boneName]
    matrix = pose_bone.matrix.copy()
    delta_position = matrix.copy().inverted()@global_position
    pose_bone.location = delta_position    
    context.view_layer.update()    
    
    matrix = pose_bone.matrix.copy()
    vectorA = matrix.copy().inverted()@pose_bone.bone.vector
    vectorB = matrix.copy().inverted()@to_armature.pose.bones[pose_bone.name].bone.vector
    R, c = getRotationScale(vectorA, vectorB)
    # pose_bone.rotation_quaternion = R
    # pose_bone.scale = (c, c, c)
    print(pose_bone.name, R, c)
    context.view_layer.update()    
    
    for childBone in pose_bone.children:
        poseBonePosition(context, childBone.name, from_armature, to_armature, mapping, global_positions)
    
def setArmaturePoseToMapping(context: bpy.types.Context, from_armature: bpy.types.Object, to_armature: bpy.types.Object, mapping: dict, mapped_positions: np.ndarray) -> None:    
    poseBonePosition(context, 'Nose', from_armature, to_armature, mapping, mapped_positions)
    
    # for pose_bone in from_armature.pose.bones:
    #     vectorA = pose_bone.bone.vector
    #     vectorB = to_armature.pose.bones[pose_bone.name].bone.vector
        
    #     R, c = getRotationScale(vectorA, vectorB)
    #     pose_bone.rotation_quaternion = R
    #     pose_bone.scale = (c, c, c)
    #     print(vectorA, vectorB)
    #     print(pose_bone.name, ' scale: ', c)

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
armature = C.view_layer.objects['Coco17']
armature_duplicate = C.view_layer.objects['Coco17_DUPLICATE']

keypoints_flat: list = getJSONKeyPoints(JSON_PATH)
camera_intrinsics: tuple = getCameraIntrinsics(CAMERA_INTRINSICS_PATH)
joints_mapping: dict = getJointsMapping(JOINTS_MAPPING_PATH)
joints_mapping_blender: dict = getJointsMapping(JOINTS_MAPPING_PATH_TO_BLENDER)

keypoints_xyz: np.ndarray = keypointsXYZ(
    keypoints_flat, DEPTH_IMAGE_PATH, camera_intrinsics)

poseTemplateReset(armature)
setDuplicateArmaturePosition(armature_duplicate, keypoints_xyz, joints_mapping_blender)
setArmaturePoseToMapping(C, armature, armature_duplicate, joints_mapping, keypoints_xyz)
