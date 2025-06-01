import bpy
import bmesh
from mathutils import Vector
import random
import math
import json


def clear_scene():
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    for obj in mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def create_building(name, location, width=4.0, depth=4.0, height=8.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    building = bpy.context.active_object
    building.name = name
    building.scale = (width, depth, height)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    building.location.z = height / 2

    bevel_modifier = building.modifiers.new(name="Bevel", type='BEVEL')
    bevel_modifier.width = 0.1
    bevel_modifier.segments = 2

    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    base_color = mat.node_tree.nodes["Principled BSDF"]
    colors = [(0.8, 0.3, 0.3, 1.0), (0.3, 0.6, 0.8, 1.0), (0.7, 0.7, 0.3, 1.0),
              (0.4, 0.7, 0.4, 1.0), (0.7, 0.4, 0.7, 1.0), (0.8, 0.5, 0.2, 1.0)]
    color_index = hash(name) % len(colors)
    base_color.inputs[0].default_value = colors[color_index]
    base_color.inputs[7].default_value = 0.2
    building.data.materials.append(mat)
    return building


def create_tree(name, location):
    trunk_height = 3.0
    trunk_radius = 0.2
    trunk_location = (location[0], location[1], trunk_height / 2)

    bpy.ops.mesh.primitive_cylinder_add(radius=trunk_radius, depth=trunk_height, location=trunk_location)
    trunk = bpy.context.active_object
    trunk.name = f"{name}_Trunk"

    trunk_mat = bpy.data.materials.new(name="Trunk_Material")
    trunk_mat.use_nodes = True
    trunk_bsdf = trunk_mat.node_tree.nodes["Principled BSDF"]
    trunk_bsdf.inputs[0].default_value = (0.4, 0.2, 0.1, 1.0)
    trunk.data.materials.append(trunk_mat)

    foliage_radius = 1.5
    foliage_location = (location[0], location[1], trunk_height)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=foliage_radius, location=foliage_location)
    foliage = bpy.context.active_object
    foliage.name = f"{name}_Foliage"

    foliage_mat = bpy.data.materials.new(name="Foliage_Material")
    foliage_mat.use_nodes = True
    foliage_bsdf = foliage_mat.node_tree.nodes["Principled BSDF"]
    foliage_bsdf.inputs[0].default_value = (0.2, 0.8, 0.3, 1.0)
    foliage_bsdf.inputs[7].default_value = 0.7
    foliage.data.materials.append(foliage_mat)

    foliage.parent = trunk
    return trunk


def create_park(name, location, radius=8.0):
    park_objects = []
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=0.1, location=location)
    park_base = bpy.context.active_object
    park_base.name = f"{name}_Base"
    park_base.location.z = 0.05

    grass_mat = bpy.data.materials.new(name="Grass_Material")
    grass_mat.use_nodes = True
    grass_bsdf = grass_mat.node_tree.nodes["Principled BSDF"]
    grass_bsdf.inputs[0].default_value = (0.2, 0.6, 0.1, 1.0)
    grass_bsdf.inputs[7].default_value = 0.8
    park_base.data.materials.append(grass_mat)
    park_objects.append(park_base)

    tree = create_tree(f"{name}_Tree", location)
    park_objects.append(tree)

    bpy.ops.object.empty_add(location=location)
    park_parent = bpy.context.active_object
    park_parent.name = name

    for obj in park_objects:
        obj.parent = park_parent
    return park_parent


def create_road_segment(name, start_pos, end_pos, width=3.0):
    center = [(start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2]
    center_z = 0.01
    length = math.dist(start_pos, end_pos)
    angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])

    bpy.ops.mesh.primitive_cube_add(size=1, location=(center[0], center[1], center_z))
    road = bpy.context.active_object
    road.name = name
    road.scale = (length, width, 0.02)
    road.rotation_euler = (0, 0, angle)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)
    bsdf.inputs[7].default_value = 0.9
    road.data.materials.append(mat)

    return road, {
        'type': 'segment',
        'start': list(start_pos),
        'end': list(end_pos),
        'center': center,
        'width': width,
        'length': length,
        'angle': angle
    }


def create_road_around_park(park_location, park_radius, road_width=3.0):
    road_size = (park_radius + road_width) * 2
    road_inner_size = park_radius * 2

    bpy.ops.mesh.primitive_plane_add(size=road_size, location=park_location)
    road_outer = bpy.context.active_object
    road_outer.name = "Road_Park_Ring"
    road_outer.location.z = 0.01

    bpy.ops.mesh.primitive_plane_add(size=road_inner_size, location=park_location)
    road_inner = bpy.context.active_object
    road_inner.name = "Road_Inner"
    road_inner.location.z = 0.01

    bool_mod = road_outer.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = road_inner
    bpy.context.view_layer.objects.active = road_outer
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.data.objects.remove(road_inner, do_unlink=True)

    mat = bpy.data.materials.new(name="Road_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)
    bsdf.inputs[7].default_value = 0.9
    road_outer.data.materials.append(mat)

    outer_radius = park_radius + road_width
    return road_outer, {
        'type': 'ring',
        'center': list(park_location),
        'inner_radius': park_radius,
        'outer_radius': outer_radius,
        'width': road_width
    }


def create_road_network(park_location, park_radius, road_width=3.0):
    roads = []
    road_info = []
    ring_road, ring_data = create_road_around_park(park_location, park_radius, road_width)
    roads.append(ring_road)
    road_info.append(ring_data)

    outer_radius = park_radius + road_width
    arterial_length = 40
    secondary_offset = 25

    arterial_roads = [
        # N
        ([park_location[0], park_location[1] + outer_radius], [park_location[0], park_location[1] + outer_radius + arterial_length]),
        # S
        ([park_location[0], park_location[1] - outer_radius], [park_location[0], park_location[1] - outer_radius - arterial_length]),
        # E (angled)
        ([park_location[0] + outer_radius, park_location[1]], [park_location[0] + outer_radius + 30, park_location[1] + 10]),
        # W (angled)
        ([park_location[0] - outer_radius, park_location[1]], [park_location[0] - outer_radius - 30, park_location[1] - 10]),
    ]

    for i, (start, end) in enumerate(arterial_roads):
        road, info = create_road_segment(f"Road_Arterial_{i + 1}", start, end, road_width)
        roads.append(road)
        road_info.append(info)

    secondary_roads = [
        # Horizontal
        ([park_location[0] - 40, park_location[1] + secondary_offset], [park_location[0] + 40, park_location[1] + secondary_offset]),
        ([park_location[0] - 40, park_location[1] - secondary_offset], [park_location[0] + 40, park_location[1] - secondary_offset]),
        # Vertical
        ([park_location[0] + secondary_offset, park_location[1] - 40], [park_location[0] + secondary_offset, park_location[1] + 40]),
        ([park_location[0] - secondary_offset, park_location[1] - 40], [park_location[0] - secondary_offset, park_location[1] + 40]),
    ]

    for i, (start, end) in enumerate(secondary_roads):
        road, info = create_road_segment(f"Road_Secondary_{i + 1}", start, end, road_width * 0.8)
        roads.append(road)
        road_info.append(info)

    return roads, road_info


def create_ground_plane(size=100):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    mat = bpy.data.materials.new(name="Ground_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = (0.4, 0.4, 0.35, 1.0)
    bsdf.inputs[7].default_value = 0.9
    ground.data.materials.append(mat)
    return ground


def generate_city_scene(num_buildings=6, park_radius=6):
    print("Generating updated city scene...")

    clear_scene()
    ground = create_ground_plane(100)

    park_location = (20, -15, 0)
    park = create_park("Park", park_location, radius=park_radius)
    print(f"Created park at {park_location} with radius {park_radius}")

    roads, road_info = create_road_network(park_location, park_radius, road_width=3.0)
    print(f"Created road network with {len(roads)} segments")

    road_data_text = bpy.data.texts.new("RoadData")
    serializable_road_info = road_info
    road_data_text.write(json.dumps(serializable_road_info, indent=2))

    buildings = []
    cluster_origin = (park_location[0] + 25, park_location[1] + 10)
    for i in range(num_buildings):
        offset_x = random.uniform(-10, 10)
        offset_y = random.uniform(-10, 10)
        x = cluster_origin[0] + offset_x
        y = cluster_origin[1] + offset_y
        building_location = (x, y, 0)
        building = create_building(f"Building_{i + 1}", building_location, width=2.0, depth=2.0, height=8.0)
        buildings.append(building)
        print(f"Created {building.name} at {building_location}")

    bpy.ops.object.light_add(type='SUN', location=(10, 10, 20))
    bpy.context.active_object.name = "Sun"
    bpy.context.active_object.data.energy = 3

    bpy.ops.object.camera_add(location=(50, -80, 60))
    camera = bpy.context.active_object
    camera.name = "Camera"
    direction = Vector((0, 0, 0)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = camera

    print("Scene setup complete.")


if __name__ == "__main__":
    generate_city_scene()
