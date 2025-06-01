import bpy
import bmesh
from mathutils import Vector
import random
import math


def clear_scene():
    """Clear all mesh objects from the scene"""
    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    # Delete them directly
    for obj in mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Also clear orphaned mesh data
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def create_building(name, location, width=4.0, depth=4.0, height=8.0):
    """Create a building with fixed dimensions"""
    # Create base cube
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    building = bpy.context.active_object
    building.name = name

    # Scale to desired dimensions
    building.scale = (width, depth, height)

    # Apply the scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Move building so it sits on ground (Z=0)
    building.location.z = height / 2

    # Add a bevel modifier for slightly rounded edges
    bevel_modifier = building.modifiers.new(name="Bevel", type='BEVEL')
    bevel_modifier.width = 0.1
    bevel_modifier.segments = 2

    # Create a simple material for the building with more distinct colors
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True

    # Set building color - make buildings more colorful and distinct
    base_color = mat.node_tree.nodes["Principled BSDF"]

    # Expanded color palette for more buildings
    colors = [
        (0.8, 0.3, 0.3, 1.0),  # Red
        (0.3, 0.6, 0.8, 1.0),  # Blue
        (0.7, 0.7, 0.3, 1.0),  # Yellow
        (0.4, 0.7, 0.4, 1.0),  # Green
        (0.7, 0.4, 0.7, 1.0),  # Purple
        (0.8, 0.5, 0.2, 1.0),  # Orange
        (0.6, 0.8, 0.8, 1.0),  # Cyan
        (0.8, 0.6, 0.4, 1.0),  # Tan
        (0.5, 0.3, 0.7, 1.0),  # Indigo
        (0.8, 0.4, 0.5, 1.0),  # Pink
        (0.4, 0.5, 0.3, 1.0),  # Olive
        (0.7, 0.5, 0.6, 1.0),  # Mauve
    ]

    # Use hash of building name to consistently assign colors
    color_index = hash(name) % len(colors)
    base_color.inputs[0].default_value = colors[color_index]
    base_color.inputs[7].default_value = 0.2  # Lower roughness for more vibrant look

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
    park_base.location.z = 0.05  # Slightly above ground

    # Create grass material
    grass_mat = bpy.data.materials.new(name="Grass_Material")
    grass_mat.use_nodes = True
    grass_bsdf = grass_mat.node_tree.nodes["Principled BSDF"]
    grass_bsdf.inputs[0].default_value = (0.2, 0.6, 0.1, 1.0)  # Green color
    grass_bsdf.inputs[7].default_value = 0.8  # Roughness
    park_base.data.materials.append(grass_mat)

    park_objects.append(park_base)

    # Create just one tree at the center of the park
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

    # Position trunk so it sits on ground
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

    # Create foliage (sphere) - position it at the TOP of the trunk
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
    center_z = 0.01  # Slightly above ground

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
    road.scale = (length, width, 0.02)  # Very thin road
    road.rotation_euler = (0, 0, angle)

    # Apply transforms
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Create road material (dark gray)
    road_mat = bpy.data.materials.new(name=f"{name}_Material")
    road_mat.use_nodes = True
    road_bsdf = road_mat.node_tree.nodes["Principled BSDF"]
    road_bsdf.inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)  # Dark gray
    road_bsdf.inputs[7].default_value = 0.9  # High roughness
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
    """Create a circular road around the park"""
    # Create outer circle
    bpy.ops.mesh.primitive_cylinder_add(
        radius=park_radius + road_width,
        depth=0.02,
        location=park_location
    )
    road_outer = bpy.context.active_object
    road_outer.name = "Road_Park_Ring"
    road_outer.location.z = 0.01

    # Create inner circle (to subtract)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=park_radius,
        depth=0.03,
        location=park_location
    )
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

    # Delete the inner object (no longer needed)
    bpy.data.objects.remove(road_inner, do_unlink=True)

    # Create road material (dark gray)
    road_mat = bpy.data.materials.new(name="Road_Material")
    road_mat.use_nodes = True
    road_bsdf = road_mat.node_tree.nodes["Principled BSDF"]
    road_bsdf.inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)  # Dark gray
    road_bsdf.inputs[7].default_value = 0.9  # High roughness
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


def create_expanded_road_network(park_location, park_radius, road_width=3.0):
    """Create a comprehensive expanded road network with many more roads"""
    roads = []
    road_info = []

    # 1. Ring road around park
    ring_road, ring_info = create_road_around_park(park_location, park_radius, road_width)
    roads.append(ring_road)
    road_info.append(ring_info)

    road_outer_radius = park_radius + road_width

    # 2. Main arterial roads extending from park (8 directions instead of 4)
    arterial_length = 50
    num_arterials = 8
    for i in range(num_arterials):
        angle = (2 * math.pi * i) / num_arterials
        start_x = road_outer_radius * math.cos(angle)
        start_y = road_outer_radius * math.sin(angle)
        end_x = (road_outer_radius + arterial_length) * math.cos(angle)
        end_y = (road_outer_radius + arterial_length) * math.sin(angle)

        start = (start_x, start_y)
        end = (end_x, end_y)

        road, info = create_road_segment(f"Road_Arterial_{i + 1}", start, end, road_width)
        roads.append(road)
        road_info.append(info)

    # 3. Secondary ring roads at different distances
    for ring_num, distance in enumerate([25, 40], 1):
        ring_points = []
        num_points = 12

        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            ring_points.append((x, y))

        # Connect consecutive points to form a ring
        for i in range(num_points):
            start = ring_points[i]
            end = ring_points[(i + 1) % num_points]
            road, info = create_road_segment(f"Road_Ring_{ring_num}_{i + 1}", start, end, road_width * 0.8)
            roads.append(road)
            road_info.append(info)

    # 4. Grid roads - horizontal and vertical
    grid_positions = [-45, -30, -15, 15, 30, 45]
    grid_extent = 60

    # Horizontal grid roads
    for i, y_pos in enumerate(grid_positions):
        start = (-grid_extent, y_pos)
        end = (grid_extent, y_pos)
        road, info = create_road_segment(f"Road_Grid_H_{i + 1}", start, end, road_width * 0.7)
        roads.append(road)
        road_info.append(info)

    # Vertical grid roads
    for i, x_pos in enumerate(grid_positions):
        start = (x_pos, -grid_extent)
        end = (x_pos, grid_extent)
        road, info = create_road_segment(f"Road_Grid_V_{i + 1}", start, end, road_width * 0.7)
        roads.append(road)
        road_info.append(info)

    # 5. Diagonal connector roads (more of them)
    diagonal_distance = 35
    diagonal_positions = [
        # Main diagonals
        ((diagonal_distance, diagonal_distance), (55, 55)),
        ((-diagonal_distance, diagonal_distance), (-55, 55)),
        ((diagonal_distance, -diagonal_distance), (55, -55)),
        ((-diagonal_distance, -diagonal_distance), (-55, -55)),
        # Additional diagonal connectors
        ((20, 35), (45, 60)),
        ((-20, 35), (-45, 60)),
        ((20, -35), (45, -60)),
        ((-20, -35), (-45, -60)),
        ((35, 20), (60, 45)),
        ((-35, 20), (-60, 45)),
        ((35, -20), (60, -45)),
        ((-35, -20), (-60, -45)),
    ]

    for i, (start, end) in enumerate(diagonal_positions):
        road, info = create_road_segment(f"Road_Diagonal_{i + 1}", start, end, road_width * 0.6)
        roads.append(road)
        road_info.append(info)

    # 6. Connecting roads between rings
    connector_angles = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi, 5 * math.pi / 4, 3 * math.pi / 2,
                        7 * math.pi / 4]
    for i, angle in enumerate(connector_angles):
        # Inner to middle ring
        start_x = 25 * math.cos(angle)
        start_y = 25 * math.sin(angle)
        end_x = 40 * math.cos(angle)
        end_y = 40 * math.sin(angle)

        road, info = create_road_segment(f"Road_Connector_{i + 1}", (start_x, start_y), (end_x, end_y),
                                         road_width * 0.5)
        roads.append(road)
        road_info.append(info)

    return roads, road_info


def create_ground_plane(size=120):
    """Create a large ground plane"""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    # Ground material
    ground_mat = bpy.data.materials.new(name="Ground_Material")
    ground_mat.use_nodes = True
    ground_bsdf = ground_mat.node_tree.nodes["Principled BSDF"]
    ground_bsdf.inputs[0].default_value = (0.4, 0.4, 0.35, 1.0)  # Grayish ground
    ground_bsdf.inputs[7].default_value = 0.9  # Roughness
    ground.data.materials.append(ground_mat)

    return ground


def generate_enhanced_city_scene(num_buildings=15, park_radius=6):
    """Generate an enhanced city scene with more buildings and extensive road network"""
    print("Generating enhanced city scene with expanded road network...")

    # Clear existing scene
    clear_scene()

    # Create ground
    ground = create_ground_plane(120)

    # Create park at center (keeping original position)
    park_location = (0, 0, 0)
    park = create_park("Park", park_location, radius=park_radius)

    print(f"Created park at {park_location} with radius {park_radius}")

    # Create expanded road network
    roads, road_info = create_expanded_road_network(park_location, park_radius, road_width=3.0)
    print(f"Created expanded road network with {len(roads)} road segments")

    # Store road info in scene for optimizer access
    road_data_text = bpy.data.texts.new("RoadData")
    import json

    # Convert road_info to JSON-serializable format
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

    # Create buildings at new strategic positions
    # Mix of different positioning strategies for more interesting layout
    buildings = []

    # Group 1: Buildings near major intersections
    intersection_positions = [
        (30, 30), (-30, 30), (30, -30), (-30, -30),  # Corner intersections
        (45, 0), (-45, 0), (0, 45), (0, -45),  # Cardinal directions
        (22, 38), (-22, 38), (22, -38), (-22, -38),  # Offset positions
    ]

    for i, pos in enumerate(intersection_positions[:12]):  # Take first 12 positions
        # Add some random variation to avoid perfect grid
        x_offset = random.uniform(-3, 3)
        y_offset = random.uniform(-3, 3)
        building_location = (pos[0] + x_offset, pos[1] + y_offset, 0)

        # Vary building dimensions slightly
        width = random.uniform(2.5, 4.0)
        depth = random.uniform(2.5, 4.0)
        height = random.uniform(6.0, 12.0)

        building = create_building(
            f"Building_{i + 1}",
            building_location,
            width=width,
            depth=depth,
            height=height
        )
        buildings.append(building)
        print(f"Created {building.name} at {building_location} (size: {width:.1f}x{depth:.1f}x{height:.1f})")

    # Group 2: Additional scattered buildings
    for i in range(12, num_buildings):
        # Random positions in outer areas, avoiding the central park area
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(20, 30)  # Keep outside the inner road network

        x = distance * math.cos(angle)
        y = distance * math.sin(angle)

        # Add some clustering by occasionally placing buildings near existing ones
        if random.random() < 0.3 and buildings:  # 30% chance to cluster
            existing_building = random.choice(buildings)
            cluster_offset_x = random.uniform(-8, 8)
            cluster_offset_y = random.uniform(-8, 8)
            x = existing_building.location.x + cluster_offset_x
            y = existing_building.location.y + cluster_offset_y

        building_location = (x, y, 0)

        # Vary building dimensions
        width = random.uniform(2.0, 5.0)
        depth = random.uniform(2.0, 5.0)
        height = random.uniform(5.0, 15.0)

        building = create_building(
            f"Building_{i + 1}",
            building_location,
            width=width,
            depth=depth,
            height=height
        )
        buildings.append(building)
        print(f"Created {building.name} at {building_location} (size: {width:.1f}x{depth:.1f}x{height:.1f})")

    # Add enhanced lighting
    # Main sun light
    bpy.ops.object.light_add(type='SUN', location=(20, 20, 30))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 4
    sun.rotation_euler = (math.radians(45), 0, math.radians(45))

    # Additional area light for softer shadows
    bpy.ops.object.light_add(type='AREA', location=(-30, -30, 25))
    area_light = bpy.context.active_object
    area_light.name = "Area_Light"
    area_light.data.energy = 2
    area_light.data.size = 10

    # Set up camera with better positioning
    bpy.ops.object.camera_add(location=(70, -70, 45))
    camera = bpy.context.active_object
    camera.name = "Camera"

    # Point camera toward the scene center
    direction = Vector((0, 0, 5)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Set camera as active
    bpy.context.scene.camera = camera

    print(f"\nEnhanced city scene generated with:")
    print(f"  - {len(buildings)} buildings with varied sizes and positions")
    print(f"  - 1 central park (unchanged position)")
    print(f"  - {len(roads)} road segments")
    print("  - Circular ring road around park")
    print("  - 8 arterial roads radiating outward")
    print("  - Multiple secondary ring roads")
    print("  - Comprehensive grid road system")
    print("  - Extensive diagonal connector roads")
    print("  - Strategic building placement near intersections")
    print("  - Enhanced lighting setup")
    print("\nThe scene now has a much more complex and realistic road network!")


# Run the enhanced generator
if __name__ == "__main__":
    generate_enhanced_city_scene(num_buildings=15, park_radius=6)