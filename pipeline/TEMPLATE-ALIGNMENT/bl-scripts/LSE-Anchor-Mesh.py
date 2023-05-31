import bpy
import bmesh

def getMesh(context: bpy.types.Context, meshname: str) -> bpy.types.Object:
    mesh_object: bpy.types.Object = context.view_layer.objects.get(meshname, None)
    mesh_data: bpy.types.Mesh = None

    if(not mesh_object):
        mesh_data = bpy.data.meshes.new(meshname)
        mesh_object = bpy.data.objects.new(meshname, mesh_data)
        C.scene.collection.children[0].objects.link(mesh_object)

    mesh_data = mesh_object.data
    bm = bmesh.new()
    bm.from_mesh(mesh_data)
    bm.free()
    return mesh_object

def createBoundaryMesh(context: bpy.types.Context, meshname: str, graph_direction_object_names: list) -> bpy.types.Object:
    mesh_object: bpy.types.Object = getMesh(context, meshname)
    mesh_data: bpy.types.Mesh = mesh_object.data
    bm = bmesh.new()
    for o_name in graph_direction_object_names:
        o: bpy.types.Object = context.view_layer.objects[o_name]
        x, y, z = o.location
        vert = bm.verts.new()
        vert.co = (x, y, z)

    bm.verts.ensure_lookup_table()

    for i, v in enumerate(bm.verts):
        i1 = (i + 1) % len(bm.verts)
        v1: bpy.types.BMVert = bm.verts[i1]
        edge: bpy.types.BMEdge = bm.edges.new((v, v1))

    bm.to_mesh(mesh_data)
    bm.free()
    return mesh_object

if __name__ == '__main__':
    anchor_mesh: bpy.types.Object = None
    C: bpy.types.Context = bpy.context
    anchor_mesh_name: str = 'LSE-Anchor-MESH'
    graph_border_connections: list = [    
        'Ear.R', 'Eye.R',
        'Nose', 
        'Eye.L', 'Ear.L', 
        'Shoulder.L', 'Elbow.L', 'Wrist.L', 
        'Hip.L', 'Knee.L', 'Ankle.L', 
        'Ankle.R', 'Knee.R', 'Hip.R',      
        'Wrist.R', 'Elbow.R', 'Shoulder.R',
    ]

    anchor_mesh = createBoundaryMesh(C, anchor_mesh_name, graph_border_connections)
