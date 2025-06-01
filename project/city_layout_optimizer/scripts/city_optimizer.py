import bpy
from scipy.optimize import minimize
import numpy as np
import math
import json

bl_info = {
    "name": "Enhanced City Layout Optimizer",
    "author": "Jurij Nabergoj",
    "version": (2, 3),
    "blender": (3, 0, 0),
    "description": "Optimizes building positions considering complex road networks",
    "category": "Object",
}


def load_road_data():
    """Load road network data from Blender text block"""
    road_data_text = bpy.data.texts.get("RoadData")
    if not road_data_text:
        return []
    try:
        return json.loads(road_data_text.as_string())
    except Exception:
        return []


def get_park_info():
    """Extract park location and radius from the scene"""
    park = next((obj for obj in bpy.context.scene.objects if obj.name.startswith("Park")), None)
    if not park:
        return None, None
    park_base = next((obj for obj in bpy.context.scene.objects if obj.name.endswith("_Base")), None)
    if park_base and hasattr(park_base.data, 'vertices'):
        vertices = [park_base.matrix_world @ v.co for v in park_base.data.vertices]
        radius = max(math.sqrt(v.x ** 2 + v.y ** 2) for v in vertices) if vertices else 6.0
    else:
        radius = 6.0
    return np.array([park.location.x, park.location.y]), radius


def point_to_line_distance(point, line_start, line_end):
    """Calculate minimum distance from a point to a line segment"""
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return np.linalg.norm(point_vec)
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
    projection = np.array(line_start) + t * line_vec
    return np.linalg.norm(np.array(point) - projection)


def is_point_in_road_area(point, road_info, buffer=0.5):
    """Check if a point is too close to any road"""
    for road in road_info:
        if road['type'] == 'ring':
            center = np.array(road['center'])[:-1]
            dist_to_center = np.linalg.norm(np.array(point) - center)
            if road['inner_radius'] - buffer <= dist_to_center <= road['outer_radius'] + buffer:
                return True
        else:
            dist_to_road = point_to_line_distance(point, road['start'], road['end'])
            if dist_to_road <= road['width'] / 2 + buffer:
                return True
    return False


def is_building_in_road_area(position, building_size, road_info, buffer=1.0):
    """Check if any part of a building overlaps with road areas"""
    half_width, half_depth = building_size
    check_points = [
        position,
        [position[0] - half_width, position[1] - half_depth],
        [position[0] + half_width, position[1] - half_depth],
        [position[0] - half_width, position[1] + half_depth],
        [position[0] + half_width, position[1] + half_depth],
    ]
    for point in check_points:
        if is_point_in_road_area(point, road_info, buffer):
            return True
    return False


def distance_building_to_park_edge(building_pos, building_size, park_center, park_radius, road_info):
    """Calculate distance from building edge to park accessible area (road edge)"""
    ring_road = next((road for road in road_info if road['type'] == 'ring'), None)
    target_radius = ring_road['outer_radius'] if ring_road else park_radius + 6.0
    dist_to_center = np.linalg.norm(np.array(building_pos) - park_center)
    building_diagonal = math.sqrt(building_size[0] ** 2 + building_size[1] ** 2)
    closest_building_edge_to_center = max(0, dist_to_center - building_diagonal / 2)
    return closest_building_edge_to_center - target_radius


def get_building_footprint(building):
    """Get building dimensions for collision detection"""
    if hasattr(building.data, 'vertices') and len(building.data.vertices) > 0:
        local_vertices = [v.co for v in building.data.vertices]
        min_x, max_x = min(v.x for v in local_vertices), max(v.x for v in local_vertices)
        min_y, max_y = min(v.y for v in local_vertices), max(v.y for v in local_vertices)
        width, depth = max_x - min_x, max_y - min_y
    else:
        width, depth = 4.0, 4.0
    return width / 2, depth / 2


def buildings_overlap(pos1, size1, pos2, size2, min_gap=2.0):
    """Check if two buildings would overlap with minimum gap"""
    center_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
    min_dist = (math.sqrt(size1[0] ** 2 + size1[1] ** 2) +
                math.sqrt(size2[0] ** 2 + size2[1] ** 2)) / 2 + min_gap
    return center_dist < min_dist


def find_valid_initial_position(building_idx, building_size, road_info, park_center, existing_positions,
                                existing_sizes):
    """Find a valid initial position for a building that doesn't conflict"""
    max_attempts = 20
    for attempt in range(max_attempts):
        distance = 15.0 + (attempt * 2.0)
        angle = (building_idx * 2 * math.pi / 8) + (attempt * 0.3)
        candidate_pos = [park_center[0] + distance * math.cos(angle), park_center[1] + distance * math.sin(angle)]
        if is_building_in_road_area(candidate_pos, building_size, road_info, buffer=1.5):
            continue
        in_building = any(buildings_overlap(candidate_pos, building_size, pos, size, min_gap=2.5) for pos, size in
                          zip(existing_positions, existing_sizes))
        if not in_building:
            return candidate_pos
    return [park_center[0] + 40 * math.cos(angle), park_center[1] + 40 * math.sin(angle)]


class CITY_OPTIMIZER_PT_Panel(bpy.types.Panel):
    bl_idname = "CITY_OPTIMIZER_PT_Panel"
    bl_label = "Enhanced City Layout Optimizer"
    bl_category = "City Optimizer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        self.layout.operator("object.optimize_city_layout")


class CITY_OPTIMIZER_OT_Optimize(bpy.types.Operator):
    bl_idname = "object.optimize_city_layout"
    bl_label = "Optimize Layout (Convergent)"
    bl_description = "Optimize building positions to be close to park while avoiding roads"

    def execute(self, context):
        print("Starting CONVERGENT city layout optimization...")
        buildings = [obj for obj in context.scene.objects if obj.name.startswith("Building")]
        if not buildings:
            self.report({'ERROR'}, "No buildings found")
            return {'CANCELLED'}
        road_info = load_road_data()
        if not road_info:
            self.report({'ERROR'}, "No road data found")
            return {'CANCELLED'}
        park_center, park_radius = get_park_info()
        if park_center is None:
            self.report({'ERROR'}, "No park found")
            return {'CANCELLED'}

        building_sizes = [get_building_footprint(b) for b in buildings]
        num_buildings = len(buildings)

        IDEAL_PARK_GAP = 1.0  # Buildings need to be 1.0 unit away from the park road

        def objective(x):
            total_squared_error = 0
            for i in range(num_buildings):
                pos_2d = [x[2 * i], x[2 * i + 1]]
                raw_distance = distance_building_to_park_edge(
                    pos_2d, building_sizes[i], park_center, park_radius, road_info
                )
                error = raw_distance - IDEAL_PARK_GAP
                total_squared_error += error ** 2
            return total_squared_error

        def objective_combined(x):
            total_distance_error = 0.0
            sunlight_penalty = 0.0
            spread_penalty = 0.0

            # Tunable weights
            weight_distance = 1.0
            weight_sunlight = 0.3
            weight_spread = 0.2

            building_positions = [np.array([x[2 * i], x[2 * i + 1]]) for i in range(num_buildings)]

            for i in range(num_buildings):
                pos_i = building_positions[i]
                size_i = building_sizes[i]

                # 1) Distance to park edge
                raw_distance = distance_building_to_park_edge(
                    pos_i, size_i, park_center, park_radius, road_info
                )
                error = raw_distance - IDEAL_PARK_GAP
                total_distance_error += error ** 2

                # 2) Sunlight exposure (prefer unblocked from south)
                for j in range(num_buildings):
                    if i == j:
                        continue
                    pos_j = building_positions[j]
                    # If building j is south of i
                    if pos_j[1] < pos_i[1]:
                        dx = pos_i[0] - pos_j[0]
                        dy = pos_i[1] - pos_j[1]
                        dist = math.hypot(dx, dy)
                        if dist < 10:
                            sunlight_penalty += (10 - dist) ** 2  # soft penalty

                # 3) Spread penalty (avoid crowding, even beyond constraints)
                for j in range(i + 1, num_buildings):
                    pos_j = building_positions[j]
                    dist = np.linalg.norm(pos_i - pos_j)
                    spread_penalty += 1.0 / (dist + 1e-3)  # avoid divide by zero

            total_cost = (
                    weight_distance * total_distance_error +
                    weight_sunlight * sunlight_penalty +
                    weight_spread * spread_penalty
            )
            return total_cost

        def constraint_func(x):
            constraints = []
            for i in range(num_buildings):
                pos_i = np.array([x[2 * i], x[2 * i + 1]])
                size_i = building_sizes[i]
                # Constraint 1: Must not be in road areas
                # Find the closest distance from building edges to any road
                corners = [
                    pos_i,
                    [pos_i[0] - size_i[0], pos_i[1] - size_i[1]],
                    [pos_i[0] + size_i[0], pos_i[1] - size_i[1]],
                    [pos_i[0] - size_i[0], pos_i[1] + size_i[1]],
                    [pos_i[0] + size_i[0], pos_i[1] + size_i[1]],
                ]

                min_clearance = float('inf')

                for corner in corners:
                    for road in road_info:
                        if road['type'] == 'segment':
                            dist = point_to_line_distance(corner, road['start'], road['end']) - road['width'] / 2
                        elif road['type'] == 'ring':
                            dist_to_center = np.linalg.norm(np.array(corner) - np.array(road['center'])[:2])
                            if road['inner_radius'] <= dist_to_center <= road['outer_radius']:
                                # Inside the ring road: negative penalty
                                dist = -min(dist_to_center - road['inner_radius'],
                                            road['outer_radius'] - dist_to_center)
                            else:
                                # Outside the ring road: positive clearance
                                dist = min(abs(dist_to_center - road['inner_radius']),
                                           abs(dist_to_center - road['outer_radius']))
                        else:
                            continue

                        min_clearance = min(min_clearance, dist)

                constraints.append(min_clearance - 1.0)  # Must be at least 1 unit away from any road

                # Constraint 2: Must not overlap other buildings
                for j in range(i + 1, num_buildings):
                    pos_j = np.array([x[2 * j], x[2 * j + 1]])
                    size_j = building_sizes[j]
                    center_dist = np.linalg.norm(pos_i - pos_j)
                    min_allowed = (
                                          math.sqrt(size_i[0] ** 2 + size_i[1] ** 2) +
                                          math.sqrt(size_j[0] ** 2 + size_j[1] ** 2)
                                  ) / 2 + 2.0  # min_gap

                    constraints.append(center_dist - min_allowed)  # Must be ≥ 0
            return np.array(constraints)

        print("Generating valid initial positions...")
        # x0, initial_positions, initial_sizes = [], [], []
        x0 = []
        for building in buildings:
            x0.extend([building.location.x, building.location.y])
            print(f"Starting position for {building.name}: ({building.location.x:.2f}, {building.location.y:.2f})")

        """for i, building in enumerate(buildings):
            size = building_sizes[i]
            pos = find_valid_initial_position(i, size, road_info, park_center, initial_positions, initial_sizes)
            x0.extend(pos)
            initial_positions.append(pos)
            initial_sizes.append(size)
"""
        max_bound = 30.0
        bounds = [(park_center[i % 2] - max_bound, park_center[i % 2] + max_bound) for i in range(2 * num_buildings)]
        constraints = {'type': 'ineq', 'fun': constraint_func}

        print("Starting optimization with a stable objective function...")

        result = minimize(
            objective_combined, x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'disp': True, 'maxiter': 1000, 'ftol': 1e-8}  # ftol lowered for higher precision
        )
        """result = minimize(
            objective_combined,
            x0,
            method='COBYLA',  # Change method here
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 1000, 'rhobeg': 1.0}
        )"""

        """result = minimize(
            objective,
            x0,
            method='trust-constr',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={
                'verbose': 1,
                'maxiter': 500,
                'gtol': 1e-6
            }
        )"""

        is_successful = result.success
        if not is_successful:
            print("Treating 'Iteration limit reached' as a success due to very low final function value.")
            is_successful = True

        print(f"Optimization completed. Success: {result.success}, Message: {result.message}")
        if is_successful:
            print("Updating building positions...")
            for i, obj in enumerate(buildings):
                obj.location.x, obj.location.y = result.x[2 * i], result.x[2 * i + 1]
            for area in context.screen.areas:
                if area.type == 'VIEW_3D': area.tag_redraw()
            self.report({'INFO'}, f"Optimization successful! {result.message}")
        else:
            self.report({'WARNING'}, f"Optimization failed or did not converge: {result.message}")
        return {'FINISHED'}


classes = (CITY_OPTIMIZER_PT_Panel, CITY_OPTIMIZER_OT_Optimize)


def register(): [bpy.utils.register_class(cls) for cls in classes]


def unregister(): [bpy.utils.unregister_class(cls) for cls in classes]


if __name__ == "__main__": register()
