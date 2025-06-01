import bpy
import bmesh
from mathutils import Vector
import random
import math


def clear_scene():
    """Clear all mesh objects from the scene"""
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    for obj in mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def create_building(name, location, width=4.0, depth=4.0, height=8.0):
    """Create a building with fixed dimensions"""
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

    colors = [
        (0.8, 0.3, 0.3, 1.0),  # Red
        (0.3, 0.6, 0.8, 1.0),  # Blue
        (0.7, 0.7, 0.3, 1.0),  # Yellow
        (0.4, 0.7, 0.4, 1.0),  # Green
        (0.7, 0.4, 0.7, 1.0),  # Purple
        (0.8, 0.5, 0.2, 1.0),  # Orange
    ]

    color_index = hash(name) % len(colors)
    base_color.inputs[0].default_value = colors[color_index]
    base_color.inputs[7].default_value = 0.2

    building.data.materials.append(mat)

    return building


def create_park(name, location, radius=8.0):
    """Create a park with trees and grass area"""
    park_objects = []

    # Create main park area (grass circle)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=0.1,
        location=location
    )
    park_base = bpy.context.active_object
    park_base.name = f"{name}_Base"
    park_base.location.z = 0.05

    # Create grass material
    grass_mat = bpy.data.materials.new(name="Grass_Material")
    grass_mat.use_nodes = True
    grass_bsdf = grass_mat.node_tree.nodes["Principled BSDF"]
    grass_bsdf.inputs[0].default_value = (0.2, 0.6, 0.1, 1.0)  # Green color
    grass_bsdf.inputs[7].default_value = 0.8
    park_base.data.materials.append(grass_mat)

    park_objects.append(park_base)

    # Create tree at the center of the park
    tree = create_tree(f"{name}_Tree", location)
    park_objects.append(tree)

    # Create a parent empty for the park
    bpy.ops.object.empty_add(location=location)
    park_parent = bpy.context.active_object
    park_parent.name = name

    # Parent all park objects to the main park object
    for obj in park_objects:
        obj.parent = park_parent

    return park_parent


def create_tree(name, location):
    """Create a simple tree with trunk and foliage properly positioned"""
    # Create trunk
    trunk_height = 3.0
    trunk_radius = 0.2

    trunk_location = (location[0], location[1], trunk_height / 2)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=trunk_radius,
        depth=trunk_height,
        location=trunk_location
    )
    trunk = bpy.context.active_object
    trunk.name = f"{name}_Trunk"

    # Trunk material (brown)
    trunk_mat = bpy.data.materials.new(name="Trunk_Material")
    trunk_mat.use_nodes = True
    trunk_bsdf = trunk_mat.node_tree.nodes["Principled BSDF"]
    trunk_bsdf.inputs[0].default_value = (0.4, 0.2, 0.1, 1.0)  # Brown
    trunk.data.materials.append(trunk_mat)

    # Create foliage (sphere) at the top of the trunk
    foliage_radius = 1.5
    foliage_location = (location[0], location[1], trunk_height)

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=foliage_radius,
        location=foliage_location
    )
    foliage = bpy.context.active_object
    foliage.name = f"{name}_Foliage"

    # Foliage material (green)
    foliage_mat = bpy.data.materials.new(name="Foliage_Material")
    foliage_mat.use_nodes = True
    foliage_bsdf = foliage_mat.node_tree.nodes["Principled BSDF"]
    foliage_bsdf.inputs[0].default_value = (0.2, 0.8, 0.3, 1.0)  # Bright green
    foliage_bsdf.inputs[7].default_value = 0.7  # Roughness
    foliage.data.materials.append(foliage_mat)

    # Parent foliage to trunk
    foliage.parent = trunk

    return trunk


def create_road_segment(name, start_pos, end_pos, width=3.0):
    """Create a single road segment between two points"""
    # Calculate road center, length and rotation
    center_x = (start_pos[0] + end_pos[0]) / 2
    center_y = (start_pos[1] + end_pos[1]) / 2
    center_z = 0.01

    length = math.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
    angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])

    # Create road segment
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(center_x, center_y, center_z)
    )
    road = bpy.context.active_object
    road.name = name

    # Scale and rotate
    road.scale = (length, width, 0.02)
    road.rotation_euler = (0, 0, angle)

    # Apply transforms
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Create road material (dark gray)
    road_mat = bpy.data.materials.new(name=f"{name}_Material")
    road_mat.use_nodes = True
    road_bsdf = road_mat.node_tree.nodes["Principled BSDF"]
    road_bsdf.inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)  # Dark gray
    road_bsdf.inputs[7].default_value = 0.9
    road.data.materials.append(road_mat)

    return road, {
        'start': start_pos,
        'end': end_pos,
        'center': (center_x, center_y),
        'width': width,
        'length': length,
        'angle': angle
    }


def create_road_around_park(park_location, park_radius, road_width=3.0):
    """Create a square road around the park"""
    # Calculate road dimensions
    road_size = (park_radius + road_width) * 2
    road_inner_size = park_radius * 2

    # Create outer square
    bpy.ops.mesh.primitive_plane_add(size=road_size, location=park_location)
    road_outer = bpy.context.active_object
    road_outer.name = "Road_Park_Ring"
    road_outer.location.z = 0.01

    # Create inner square (to subtract)
    bpy.ops.mesh.primitive_plane_add(size=road_inner_size, location=park_location)
    road_inner = bpy.context.active_object
    road_inner.name = "Road_Inner"
    road_inner.location.z = 0.01

    # Use boolean modifier to create road shape
    bool_modifier = road_outer.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_modifier.operation = 'DIFFERENCE'
    bool_modifier.object = road_inner

    # Apply the modifier
    bpy.context.view_layer.objects.active = road_outer
    bpy.ops.object.modifier_apply(modifier="Boolean")

    bpy.data.objects.remove(road_inner, do_unlink=True)

    # Create road material (dark gray)
    road_mat = bpy.data.materials.new(name="Road_Material")
    road_mat.use_nodes = True
    road_bsdf = road_mat.node_tree.nodes["Principled BSDF"]
    road_bsdf.inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)  # Dark gray
    road_bsdf.inputs[7].default_value = 0.9
    road_outer.data.materials.append(road_mat)

    # Return road info for the optimizer
    road_outer_radius = park_radius + road_width
    return road_outer, {
        'type': 'ring',
        'center': park_location,
        'inner_radius': park_radius,
        'outer_radius': road_outer_radius,
        'width': road_width
    }


def create_road_network(park_location, park_radius, road_width=3.0):
    """Create a comprehensive road network"""
    roads = []
    road_info = []

    # 1. Ring road around park
    ring_road, ring_info = create_road_around_park(park_location, park_radius, road_width)
    roads.append(ring_road)
    road_info.append(ring_info)

    road_outer_radius = park_radius + road_width

    # 2. Main arterial roads extending from park (4 directions)
    arterial_length = 40
    arterial_roads = [
        # North
        ((0, road_outer_radius), (0, road_outer_radius + arterial_length)),
        # South
        ((0, -road_outer_radius), (0, -(road_outer_radius + arterial_length))),
        # East
        ((road_outer_radius, 0), (road_outer_radius + arterial_length, 0)),
        # West
        ((-road_outer_radius, 0), (-(road_outer_radius + arterial_length), 0))
    ]

    for i, (start, end) in enumerate(arterial_roads):
        road, info = create_road_segment(f"Road_Arterial_{i + 1}", start, end, road_width)
        roads.append(road)
        road_info.append(info)

    # 3. Secondary connecting roads (forming a grid)
    secondary_distance = 20
    secondary_roads = [
        # Horizontal roads
        ((-30, secondary_distance), (30, secondary_distance)),
        ((-30, -secondary_distance), (30, -secondary_distance)),
        # Vertical roads
        ((secondary_distance, -30), (secondary_distance, 30)),
        ((-secondary_distance, -30), (-secondary_distance, 30))
    ]

    for i, (start, end) in enumerate(secondary_roads):
        road, info = create_road_segment(f"Road_Secondary_{i + 1}", start, end, road_width * 0.8)
        roads.append(road)
        road_info.append(info)

    # 4. Diagonal connector roads
    diagonal_roads = [
        # NE diagonal
        ((road_outer_radius * 0.7, road_outer_radius * 0.7), (25, 25)),
        # NW diagonal
        ((-road_outer_radius * 0.7, road_outer_radius * 0.7), (-25, 25)),
        # SE diagonal
        ((road_outer_radius * 0.7, -road_outer_radius * 0.7), (25, -25)),
        # SW diagonal
        ((-road_outer_radius * 0.7, -road_outer_radius * 0.7), (-25, -25))
    ]

    for i, (start, end) in enumerate(diagonal_roads):
        road, info = create_road_segment(f"Road_Diagonal_{i + 1}", start, end, road_width * 0.6)
        roads.append(road)
        road_info.append(info)

    return roads, road_info


def create_ground_plane(size=100):
    """Create a large ground plane"""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    # Ground material
    ground_mat = bpy.data.materials.new(name="Ground_Material")
    ground_mat.use_nodes = True
    ground_bsdf = ground_mat.node_tree.nodes["Principled BSDF"]
    ground_bsdf.inputs[0].default_value = (0.4, 0.4, 0.35, 1.0)
    ground_bsdf.inputs[7].default_value = 0.9
    ground.data.materials.append(ground_mat)

    return ground


def generate_city_scene(num_buildings=8, park_radius=6):
    """Generate a complete city scene with comprehensive road network"""
    print("Generating enhanced city scene with road network...")

    # Clear existing scene
    clear_scene()

    # Create ground
    ground = create_ground_plane(100)

    # Create park at center
    park_location = (0, 0, 0)
    park = create_park("Park", park_location, radius=park_radius)

    print(f"Created park at {park_location} with radius {park_radius}")

    # Create road network
    roads, road_info = create_road_network(park_location, park_radius, road_width=3.0)
    print(f"Created road network with {len(roads)} road segments")

    # Store road info in scene for optimizer access
    road_data_text = bpy.data.texts.new("RoadData")
    import json

    serializable_road_info = []
    for info in road_info:
        if info.get('type') == 'ring':
            serializable_road_info.append({
                'type': 'ring',
                'center': list(info['center']),
                'inner_radius': info['inner_radius'],
                'outer_radius': info['outer_radius'],
                'width': info['width']
            })
        else:
            serializable_road_info.append({
                'type': 'segment',
                'start': list(info['start']),
                'end': list(info['end']),
                'center': list(info['center']),
                'width': info['width'],
                'length': info['length'],
                'angle': info['angle']
            })

    road_data_text.write(json.dumps(serializable_road_info, indent=2))
    print("Road data stored in Blender text block 'RoadData'")

    # Place buildings in a rough circle around the road network
    initial_distance = 30  # Distance from park center
    buildings = []

    for i in range(num_buildings):
        angle = (2 * math.pi * i) / num_buildings + math.pi / 6
        x = initial_distance * math.cos(angle)
        y = initial_distance * math.sin(angle)

        building_location = (x, y, 0)

        # All buildings have same dimensions
        building = create_building(
            f"Building_{i + 1}",
            building_location,
            width=2.0,  # Fixed width
            depth=2.0,  # Fixed depth
            height=8.0  # Fixed height
        )
        buildings.append(building)

        print(f"Created {building.name} at {building_location} (size: 4x4x8)")

    # Lighting
    bpy.ops.object.light_add(type='SUN', location=(10, 10, 20))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3

    # Camera
    bpy.ops.object.camera_add(location=(50, -80, 60))
    camera = bpy.context.active_object
    camera.name = "Camera"

    # Point camera toward the scene center
    direction = Vector((0, 0, 0)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = camera

    print(f"City scene generated with:")
    print(f"  - {len(buildings)} buildings")
    print(f"  - 1 central park")
    print(f"  - {len(roads)} road segments")
    print("  - Ring road around park")
    print("  - Arterial roads in 4 directions")
    print("  - Secondary grid roads")
    print("  - Diagonal connector roads")


if __name__ == "__main__":
    generate_city_scene(num_buildings=8, park_radius=6)