import bpy
import math
import pathlib
import json

import numpy as np
import cv2

from mathutils import Vector, Matrix, Quaternion

from GenericMarkerCreator28.utils.meshmathutils import umeyama

# def getcRT(from_positions: np.ndarray, to_positions: np.ndarray)->tuple:
    

def getRotationScale(vectorA: Vector, vectorB: Vector)->tuple:
    scale: float = vectorB.length / vectorA.length
    return vectorA.rotation_difference(vectorB), scale


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

def poseTemplateReset(armature_ob: bpy.types.Object)->None:
    for pose_bone in armature_ob.pose.bones:
        pose_bone.rotation_quaternion = Quaternion((0, 0, 0), 1)
        pose_bone.scale = (1, 1, 1)
        pose_bone.location = (0, 0, 0)

def setDuplicateArmaturePosition(armature_object: bpy.types.Object, positions: np.ndarray, mapping_to_blender: dict):
    print(armature_object.hide_get())
    if(armature_object.hide_get()):
        return
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
        if(edit_bone.parent):
            edit_bone.head = edit_bone.parent.tail
    
    bpy.ops.object.mode_set(mode='OBJECT')

def poseBonePosition(context: bpy.types.Context, boneName: str, from_armature: bpy.types.Object, position: np.ndarray)->None:
    def getMatrix(bone):
        edit_bone = bone
        m = edit_bone.matrix
        return m

    position = Vector(position)
    pose_bone = from_armature.pose.bones[boneName]
    matrix = getMatrix(pose_bone)
    delta = matrix.inverted()@position
    pose_bone.location = delta
    C.scene.cursor.location = position

def setArmaturePoseToMapping(context:bpy.types.Context, from_armature: bpy.types.Object, mapping: dict, mapped_positions: np.ndarray)->None:
    for pose_bone in from_armature.pose.bones:
        boneName: str = pose_bone.name
        mapped_position: np.ndarray = mapped_positions[mapping[boneName]['mapped']]
        poseBonePosition(context, boneName, from_armature, mapped_position)

JSON_FILE_NAME: str = 'test4160_keypoints.json'
JSON_PATH: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('tests', JSON_FILE_NAME)
CAMERA_INTRINSICS_PATH: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('tests', 'camera_intrinsics.npy')
DEPTH_IMAGE_PATH: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('tests', '4160-DEPTH.png')
JOINTS_MAPPING_PATH: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('blender-to-coco-mapping.json')
JOINTS_MAPPING_PATH_TO_BLENDER: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('coco-to-blender-mapping.json')

C = bpy.context
armature = C.view_layer.objects['Coco17']
armature_duplicate = C.view_layer.objects['Coco17_DUPLICATE']

keypoints_flat = getJSONKeyPoints(JSON_PATH)
camera_intrinsics = getCameraIntrinsics(CAMERA_INTRINSICS_PATH)
joints_mapping = getJointsMapping(JOINTS_MAPPING_PATH)
joints_mapping_blender = getJointsMapping(JOINTS_MAPPING_PATH_TO_BLENDER)

keypoints_xyz = keypointsXYZ(keypoints_flat, DEPTH_IMAGE_PATH, camera_intrinsics)
nose_position = keypoints_xyz[joints_mapping['Nose']['mapped']]
eye_l_position = keypoints_xyz[joints_mapping['Eye.L']['mapped']]
eye_r_position = keypoints_xyz[joints_mapping['Eye.R']['mapped']]
ear_l_position = keypoints_xyz[joints_mapping['Ear.L']['mapped']]
ear_r_position = keypoints_xyz[joints_mapping['Ear.R']['mapped']]

poseTemplateReset(armature)
# setDuplicateArmaturePosition(armature_duplicate, keypoints_xyz, joints_mapping_blender)
setArmaturePoseToMapping(C, armature, joints_mapping, keypoints_xyz)