#EXAMPLE RUN FROM TERMINAL 
#blender312 ./TEMPLATE-ALIGNMENT/blends/Keypoints-Fitting-MediaPipe-LSE.blend -b --python ./TEMPLATE-ALIGNMENT/bl-scripts/Keypoints-Fitting-LSE.py -- -dimg ./assets/Synthetic/DEPTH/1-DEPTH.png -kpjs output/Synthetic/mediapipe_out/1-RGB_keypoints.json
import bpy, bmesh

import sys
import time
import logging
import pathlib
import json
import argparse

import numpy as np
import cv2

import mathutils
from mathutils import Vector, Matrix, Quaternion

CONFIG_COCO: str = 'coco'
CONFIG_BLAZEPOSE: str = 'blazepose'

class CustomLoggingFormatter(logging.Formatter):
    grey = "\\x1b[38;21m"
    yellow = "\\x1b[33;21m"
    red = "\\x1b[31;21m"
    bold_red = "\\x1b[31;1m"
    reset = "\\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class TimeLog():
    _messages: list[str] = []
    _timings: list[float] = []
    
    def __init__(self) -> None:
        pass
    
    def addMessage(self, msg: str, timeconsumed: float)->None:
        index: int = len(self._messages) + 1 
        self._messages.append(f'Line {index}: {msg}: {timeconsumed}')
        self._timings.append(timeconsumed)
    
    @property
    def messages(self)->list[str]:
        return self._messages
    
    def __repr__(self) -> str:
        messages: list[str] = self._messages[:]
        total_time: float = sum(self._timings)
        messages.append(f'Line {len(messages) + 1}: Total Time: {total_time}')
        return '\n'.join(messages)

    def __str__(self) -> str:
        return self.__repr__()
    
class VertexGroup():
    vertices: list = []
    count: int = 0
    name: str = None

    def __init__(self) -> None:
        self.vertices = []
        self.count = 0

    def addVertexIndex(self, index: int)->None:
        self.vertices.append(index)
        self.count = len(self.vertices)

    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Name: {self.name}, Count: #{self.count}'


class AlignmentArguments():
    lse_config: str = CONFIG_BLAZEPOSE
    camera_intrinsics_path: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('../').joinpath('config').joinpath('camera_intrinsics.npy')
    keypoints_to_blender_path: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('../').joinpath('config').joinpath('blazepose-to-blender-mapping.json')
    blender_to_keypoints_path: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('../').joinpath('config').joinpath('blender-to-blazepose-mapping.json')
    rgb_image_path: pathlib.Path = None
    segmentation_image_path: pathlib.Path = None
    depth_image_path: pathlib.Path = None
    keypoints_json_path: pathlib.Path = None

    def __init__(self, system_arguments: list[str]):        
        self._processArguments(system_arguments)
    
    def _processArguments(self, system_arguments)->None:
        user_arguments = ''
        if '--' in system_arguments:
            user_arguments = system_arguments[system_arguments.index('--') + 1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('-lsetp', '--lse-type', dest='lsetype', type=str, default=self.lse_config)
        parser.add_argument('-ci', '--camera-intrinsics', dest='ci', type=pathlib.Path, default=self.camera_intrinsics_path)        
        parser.add_argument('-dimg', '--depth-image', dest='depth', type=pathlib.Path)
        parser.add_argument('-rimg', '--rgb-image', dest='rgb', type=pathlib.Path)
        parser.add_argument('-simg', '--seg-image', dest='segmentation', type=pathlib.Path)
        parser.add_argument('-kpjs', '--keypoints-json', dest='keypoints_json', type=pathlib.Path)

        args = parser.parse_known_args(user_arguments)[0]        

        self.lse_config = args.lsetype
        self.camera_intrinsics_path = args.ci
        self.rgb_image_path = args.rgb
        self.segmentation_image_path = args.segmentation
        self.depth_image_path = args.depth
        self.keypoints_json_path = args.keypoints_json

        if(args.lsetype == CONFIG_COCO):
            self.blender_to_keypoints_path = pathlib.Path(bpy.path.abspath('//')).joinpath('config').joinpath('blender-to-coco-mapping.json')
            self.keypoints_to_blender_path = pathlib.Path(bpy.path.abspath('//')).joinpath('config').joinpath('coco-to-blender-mapping.json')
        
        if(self.rgb_image_path):
            self.rgb_image_path = self.rgb_image_path.resolve()
        if(self.segmentation_image_path):
            self.segmentation_image_path = self.segmentation_image_path.resolve()
        if(self.depth_image_path):
            self.depth_image_path = self.depth_image_path.resolve()            
        if(self.keypoints_json_path):
            self.keypoints_json_path = self.keypoints_json_path.resolve()

    

class PointcloudRegistration():
    fx: float
    fy: float
    cx: float
    cy: float

    context: bpy.types.Context = bpy.context
    _template_mesh: bpy.types.Object = None

    alignment_arguments: AlignmentArguments = None

    joints_mapping: dict
    joints_mapping_blender: dict

    rgb_image: np.ndarray
    segmentation_image: np.ndarray
    depth_image: np.ndarray
    
    max_depth: float
    keypoints_xyz: np.ndarray

    vertex_group_lookup: dict
    logger: logging.RootLogger = logging.getLogger("")
    
    timelogger: TimeLog = TimeLog()

    def __init__(self):        

        self._template_mesh = self.context.view_layer.objects['Human_clean']
        self.vertex_group_lookup = self._create_vertex_group_table(self._template_mesh)

        self.alignment_arguments = AlignmentArguments(sys.argv)
        self._setLoggingProceedure()

        if(not self.alignment_arguments.depth_image_path or not self.alignment_arguments.keypoints_json_path):
            raise ValueError('Need both depth and 2d keypoints to register the pointcloud')
        
        if(not self.alignment_arguments.rgb_image_path or not self.alignment_arguments.segmentation_image_path):
            raise ValueError('Need both RGB and PERSON SEGMENTATION image to register the pointcloud')
        
        self.fx, self.fy, self.cx, self.cy = self._getCameraIntrinsics(self.alignment_arguments.camera_intrinsics_path) 
        self.joints_mapping = self._getJointsMapping(self.alignment_arguments.blender_to_keypoints_path)
        self.joints_mapping_blender = self._getJointsMapping(self.alignment_arguments.keypoints_to_blender_path)
        
        self.rgb_image = cv2.imread(f'{self.alignment_arguments.rgb_image_path}', cv2.IMREAD_UNCHANGED)[:,:,:3]
        self.segmentation_image = cv2.imread(f'{self.alignment_arguments.segmentation_image_path}', cv2.IMREAD_UNCHANGED)[:,:,:3]        
        self.depth_image, self.max_depth = self._getDepthImage(self.alignment_arguments.depth_image_path)
        
        self.keypoints_xyz = self._getKeypoints3D(
            self._getKeypointsFlat(self.alignment_arguments.keypoints_json_path), (self.fx, self.fy, self.cx, self.cy), 
            self.depth_image, self.max_depth
            )
    
        self._register()

    # Get the bmesh mesh data, it is the job of the user to call bm.free after the operations are finished

    def _getBMMesh(self)->bmesh.types.BMesh:
        bm = bmesh.new()
        bm.from_mesh(self._template_mesh.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        return bm
    
    def _update_evaluate_template_mesh(self)->bpy.types.Object:
        depsgraph = self.context.evaluated_depsgraph_get()
        depsgraph.update()        
        #Get the evaluated mesh based on the depsgraph for correct positions
        mesh = self._template_mesh.evaluated_get(depsgraph)
        return mesh

    def _setLoggingProceedure(self)->None:
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)
        stdout.setFormatter(CustomLoggingFormatter())
        self.logger.addHandler(stdout)

    def _getMeanPosition(self, vertex_indices: list)->tuple:
        mean_positions = np.ndarray((len(vertex_indices), 3))
        evaluated_mesh = self.evaluated_mesh
        for i, vid in enumerate(vertex_indices):
            vertex = evaluated_mesh.data.vertices[vid]
            mean_positions[i] = (evaluated_mesh.matrix_world@vertex.co).to_tuple()
        x, y, z = mean_positions.mean(axis=0)
        return (x, y, z), mean_positions
    
    #mesh argument should be depsgraph evaluated mesh
    def _create_vertex_group_table(self, mesh)->dict:
        table: dict = {}    
        for vg in mesh.vertex_groups:
            vgroup = VertexGroup()
            vgroup.name = vg.name
            for v in mesh.data.vertices:
                gids = [g.group for g in v.groups]
                if(vg.index in gids):
                    weight = vg.weight(v.index)
                    if(weight > (1.0 - 1e-2)):
                        vgroup.addVertexIndex(v.index)
            table[vg.name] = vgroup
        return table

    # Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
    # with least-squares error.
    # Returns (scale factor c, rotation matrix R, translation vector t) such that
    #   Q = P*cR + t
    # if they align perfectly, or such that
    #   SUM over point i ( | P_i*cR + t - Q_i |^2 )
    # is minimised if they don't align perfectly.
    def _umeyama(self, P: np.ndarray, Q: np.ndarray):
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
    
    def _getJointsMapping(self, joints_json_path: pathlib.Path) -> dict:
        f = open(joints_json_path)
        joints_json = json.load(f)
        f.close()
        return joints_json

    def _getCameraIntrinsics(self, camera_intrinsics_path: pathlib.Path) -> tuple:
        camera_intrinsics: np.ndarray = np.load(camera_intrinsics_path)
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2].astype(
            float), camera_intrinsics[1, 2].astype(float)

        return fx, fy, cx, cy

    def _getKeypointsFlat(self, json_path: pathlib.Path) -> list:
        f = open(json_path)
        annotations = json.load(f)
        f.close()
        return annotations['annotations']

    def _getDepthImage(self, depth_image_path: pathlib.Path) -> tuple:
        # load depth image using OpenCV - can be replaced by any other library that loads image to numpy array
        depth_im = cv2.imread(f'{depth_image_path}', -1)
        # Get the depth values from the depth image using the indices
        raveled_depth = depth_im.ravel()
        max_depth = np.max(raveled_depth)
        return depth_im, max_depth
    
    def _d2ToD3(self, x: int, y: int)->tuple:
        z = self.depth_image[y, x] / self.max_depth
        x3D, y3D, z3D = ((x - self.cx) * z) / self.fx, ((y - self.cy) * z) / self.fy, z        
        return x3D, y3D, z3D
        
        
    def _getKeypoints3D(self, flat_keypoints: list, camera_intrinsics: tuple, nd_depth_img: np.ndarray, max_depth: float) -> np.ndarray:
        fx, fy, cx, cy = camera_intrinsics
        depth_image = nd_depth_img
        height, width = depth_image.shape

        xyz_keypoints = []
        plane_points = [Vector((0, 0, 1)), Vector((width, 0, 1)), Vector((width, height, 1)), Vector((0, height, 1))]
        for i in range(0, len(flat_keypoints), 3):
            x, y = int(flat_keypoints[i]), int(flat_keypoints[i + 1])
            # z = depth_image[y, x] / max_depth
            # x3D, y3D, z3D = ((x - cx) * z) / fx, ((y - cy) * z) / fy, z
            x3D, y3D, z3D = self._d2ToD3(x, y)            
            xyz_keypoints.append([x3D, y3D, z3D])
        
        for p_point in plane_points:
            x, y, z = p_point
            x3D, y3D, z3D = ((x - cx) * z) / fx, ((y - cy) * z) / fy, z
        
        return np.array(xyz_keypoints)
        
    def _rigidAlign(self, keypoint_xyz_positions: np.ndarray)->None:        
        hook_positions = np.zeros((len(self.joints_mapping.keys()), 3))
        target_positions = np.zeros((keypoint_xyz_positions.shape[0], 3))
        mapping: dict = self.joints_mapping

        for vgkey in self.vertex_group_lookup.keys():
            vgroup = self.vertex_group_lookup[vgkey]
                    
            if(mapping.get(vgroup.name, None)):
                index: int = mapping[vgroup.name]['mapped']
                position: np.ndarray = keypoint_xyz_positions[index]
                hook_positions[index], _ = self._getMeanPosition(vgroup.vertices)
                target_positions[index] = position

        c, R, T = umeyama(hook_positions, target_positions)
        rotation_quat = Matrix(R).to_quaternion()
        rotation_quat.w *= -1

        self._template_mesh.rotation_mode = 'QUATERNION'
        self._template_mesh.rotation_quaternion = -rotation_quat
        self._template_mesh.scale = (c, c, c)
        self._template_mesh.location = Vector(T)

        # bpy.ops.object.select_all(action="DESELECT")
        # self._template_mesh.select_set(True)
        # self.context.view_layer.objects.active = self._template_mesh
        # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        self.context.view_layer.update() 
    
    def _nonRigidAlign(self, keypoint_xyz_positions: np.ndarray)->None:
        mapping: dict = self.joints_mapping
        self._template_mesh.modifiers.clear()
        bpy.ops.object.select_all(action="DESELECT")
        self.context.view_layer.objects.active = self._template_mesh
        bpy.ops.object.mode_set(mode='EDIT')

        for vgkey in self.vertex_group_lookup.keys():
            bpy.ops.mesh.select_all(action="DESELECT")
            vgroup: VertexGroup = self.vertex_group_lookup[vgkey]
            
            self._template_mesh.vertex_groups.active_index = self._template_mesh.vertex_groups.get(vgroup.name).index
            bpy.ops.object.vertex_group_select()
            bpy.ops.object.hook_add_newob()            
            
            hook_object: bpy.types.Object = self.context.selected_objects[0]
            hook_object.name = vgroup.name
            hook_object.empty_display_size = 1e-3
            
            modifier = self._template_mesh.modifiers[-1]
            modifier.name = vgroup.name
            modifier.vertex_group = vgroup.name            
        
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action="DESELECT")      
        # self.context.view_layer.update()
        
        lse_vgroup = self._template_mesh.vertex_groups.new(name='LSE')
        lse_vertices = []
        for vgkey in self.vertex_group_lookup.keys():
            vgroup: VertexGroup = self.vertex_group_lookup[vgkey]
            lse_vertices.extend(vgroup.vertices)
        
        
        lse_vgroup.add(lse_vertices, weight=1.0, type='REPLACE')

        lse_mod = self._template_mesh.modifiers.new('LSE-Deform', type='LAPLACIANDEFORM')
        lse_mod.iterations = 1
        lse_mod.vertex_group = lse_vgroup.name
        bpy.ops.object.laplaciandeform_bind({"object" : self._template_mesh}, modifier=lse_mod.name)

        for vgkey in self.vertex_group_lookup.keys():
            vgroup: VertexGroup = self.vertex_group_lookup[vgkey]
            hook: bpy.types.Object = self.context.view_layer.objects.get(vgroup.name)
            index: int = mapping[vgroup.name]['mapped']
            position: np.ndarray = keypoint_xyz_positions[index]
            hook.location = position
            
        
        # for vgkey in self.vertex_group_lookup.keys():
        #     mod = self._template_mesh.modifiers.get(vgkey)
        #     bpy.ops.object.modifier_apply(modifier=mod.name)
            
        # bpy.ops.object.modifier_apply(modifier=lse_mod.name)
        # self._template_mesh.modifiers.clear()

        # self.context.view_layer.update() 


    #Step 3 of the method - alignment of avatar with the depth map using uplifted joint positions in 3D
    def _templateFitting(self)->None:
        
        #Step 3.1: First perform the rigid alignment
        ralign_start: float = time.time()
        self._rigidAlign(self.keypoints_xyz)
        ralign_end: float = time.time()
        
        self.timelogger.addMessage('Time for rigid alignment(Step 3.1): ', (ralign_end - ralign_start))

        #Step 3.2: Then perform the non-rigid alignment using LSE
        nralign_start: float = time.time()
        self._nonRigidAlign(self.keypoints_xyz)
        nralign_end: float = time.time()
        self.timelogger.addMessage('Time for non-rigid alignment(Step 3.2): ', (nralign_end - nralign_start))
    
    def _surfaceRegistration(self)->None:
        time_start = time.time()
        bpy_texture_image = bpy.data.images.get('template-Human_clean-uv-layout.png')
        uv_image = cv2.imread(bpy.path.abspath(bpy_texture_image.filepath), cv2.IMREAD_UNCHANGED)[:,:,:3]
                
        depsgraph = self.context.evaluated_depsgraph_get()
        depsgraph.update()        
        
        mesh_bvh = mathutils.bvhtree.BVHTree.FromObject(self._template_mesh, depsgraph=depsgraph)
        polygons = self._template_mesh.data.polygons
        uvlayerdata = self._template_mesh.data.uv_layers.get('UVMap').data
        
        
        rgb_image_output = np.copy(self.rgb_image)
        non_black_pixels_mask = np.any(self.segmentation_image != [0, 0, 0], axis=-1)  
        rows, columns = np.where(non_black_pixels_mask != 0)
        
        for y, x in zip(rows, columns):
            x3D, y3D, z3D = self._d2ToD3(x, y)
            local_coords = self._template_mesh.matrix_world.copy().inverted()@Vector((x3D, y3D, z3D))
            _, _, fid, _ = mesh_bvh.find_nearest(local_coords)
            face = polygons[fid]
            uvs = np.zeros((len(face.loop_indices), 2))
            for i, lid in enumerate(face.loop_indices):
                u, v = uvlayerdata[lid].uv
                uvs[i] = (u, v)
            u, v = uvs.mean(axis=0)            
            uv_x = int(u * uv_image.shape[1])
            uv_y = int((1 - v) * uv_image.shape[0])
            
            uv_rgb = uv_image[uv_y, uv_x]
            current_rgb = self.rgb_image[y, x]
            
            rgb_image_output[y, x] = ((uv_rgb * 0.6) + (current_rgb * 0.4)).astype(int)
        
        img_name: str = self.alignment_arguments.keypoints_json_path.stem
        img_file_path: pathlib.Path = self.alignment_arguments.keypoints_json_path.parent.joinpath(f'{img_name}.jpg')
        cv2.imwrite(f'{img_file_path}', rgb_image_output)
        time_end = time.time()
        self.timelogger.addMessage('Time for registration(Step 4): ', time_end - time_start)
            
    

    def _register(self):

        #Blender starts from step 3 of the pipeline 
        #Step 3: Align avatar with the depth map       
        self._templateFitting()

        #Step 4: Registration of depth map on the template 
        self._surfaceRegistration()
        
        #Step 5: 2D Surface Reconstruction
        #Step 6: Volume estimation
        #Step 7: Respiratory analysis - probably will be done outside Blender

        blend_name: str = self.alignment_arguments.keypoints_json_path.stem
        # blend_file_path: pathlib.Path = self.alignment_arguments.keypoints_json_path.parent.joinpath(f'{blend_name}.blend')
        # print('SAVE BLEND FILE TO ', blend_file_path)
        # bpy.ops.wm.save_as_mainfile(filepath=f'{blend_file_path}')
        log_file_path: pathlib.Path = self.alignment_arguments.keypoints_json_path.parent.joinpath(f'{blend_name}.log')
        f = open(f'{log_file_path}', 'w')
        f.write(f'{self.timelogger}')
        f.close()
        sys.exit(0)
    
    @property
    def evaluated_mesh(self)->bpy.types.Object:
        return self._update_evaluate_template_mesh()

    @property
    def template_mesh(self)->bpy.types.Object:
        return self.evaluated_mesh


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
    bpy.path.abspath('//')).joinpath('blender-to-blazepose-mapping.json')
JOINTS_MAPPING_PATH_TO_BLENDER: pathlib.Path = pathlib.Path(
    bpy.path.abspath('//')).joinpath('blazepose-to-blender-mapping.json')


if __name__ == '__main__':

    C = bpy.context
    mesh = C.view_layer.objects['Human_clean']
    registration_tool: PointcloudRegistration = PointcloudRegistration()
    # keypoints_flat: list = getJSONKeyPoints(JSON_PATH)
    # camera_intrinsics: tuple = getCameraIntrinsics(CAMERA_INTRINSICS_PATH)
    # joints_mapping: dict = getJointsMapping(JOINTS_MAPPING_PATH)
    # joints_mapping_blender: dict = getJointsMapping(JOINTS_MAPPING_PATH_TO_BLENDER)

    # keypoints_xyz: np.ndarray = keypointsXYZ(
    #     keypoints_flat, DEPTH_IMAGE_PATH, camera_intrinsics)

    # doLSE(C, mesh, joints_mapping, keypoints_xyz)
    # bpy.ops.wm.save_as_mainfile(filepath=bpy.path.abspath(f'//{JSON_PATH.parent.name}/{JSON_PATH.stem}.blend'))






# for vgkey in self.vertex_group_lookup.keys():
        #     vgroup: VertexGroup = self.vertex_group_lookup[vgkey]
        #     bl_vgroup = self._template_mesh.vertex_groups.get(vgroup.name, None)
        #     mean_position: tuple = self._getMeanPosition(vgroup.vertices)
        #     bpy.ops.object.select_all(action="DESELECT")
        #     bpy.ops.object.empty_add(type='SPHERE')
        #     hook_object: bpy.types.Object = self.context.selected_objects[0]
        #     hook_object.name = vgroup.name
        #     hook_object.location = mean_position
        #     hook_object.empty_display_size = 1e-3

        #     modifier = self._template_mesh.modifiers.new(vgroup.name, type='HOOK')
        #     modifier.vertex_group = vgroup.name
        #     modifier.object = hook_object