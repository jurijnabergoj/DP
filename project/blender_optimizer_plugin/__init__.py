import math
import bpy
import numpy as np
from mathutils import Vector
from scipy.optimize import minimize

from .scene_utils import load_road_data, get_park_info, get_building_footprint
from .optimizer import objective_combined
from .geometry_utils import point_to_line_distance, buildings_overlap

bl_info = {
    "name": "City Layout Optimizer",
    "author": "Jurij Nabergoj",
    "version": (2, 4),
    "blender": (3, 0, 0),
    "description": "Optimizes building positions considering complex road networks",
    "category": "Object",
}


class CITY_OPTIMIZER_PT_Panel(bpy.types.Panel):
    bl_idname = "CITY_OPTIMIZER_PT_Panel"
    bl_label = "City Layout Optimizer"
    bl_category = "City Optimizer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        self.layout.operator("object.optimize_city_layout")


class CITY_OPTIMIZER_OT_Optimize(bpy.types.Operator):
    bl_idname = "object.optimize_city_layout"
    bl_label = "Optimize Layout"
    bl_description = "Optimize building positions to be close to park while avoiding roads"

    def execute(self, context):
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
        IDEAL_PARK_GAP = 1.0

        # Initial positions
        x0 = []
        for building in buildings:
            x0.extend([building.location.x, building.location.y])

        # Bounds
        max_bound = 30.0
        bounds = [
            (park_center[i % 2] - max_bound, park_center[i % 2] + max_bound)
            for i in range(2 * num_buildings)
        ]

        # Constraints
        def constraint_func(x):
            constraints = []
            for i in range(num_buildings):
                pos_i = np.array([x[2 * i], x[2 * i + 1]])
                size_i = building_sizes[i]

                # Corner clearance to roads
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
                                dist = -min(dist_to_center - road['inner_radius'],
                                            road['outer_radius'] - dist_to_center)
                            else:
                                dist = min(abs(dist_to_center - road['inner_radius']),
                                           abs(dist_to_center - road['outer_radius']))
                        else:
                            continue
                        min_clearance = min(min_clearance, dist)

                constraints.append(min_clearance - 1.0)

                # Non-overlap constraint
                for j in range(i + 1, num_buildings):
                    pos_j = np.array([x[2 * j], x[2 * j + 1]])
                    size_j = building_sizes[j]
                    center_dist = np.linalg.norm(pos_i - pos_j)
                    min_allowed = (
                                          math.sqrt(size_i[0] ** 2 + size_i[1] ** 2) +
                                          math.sqrt(size_j[0] ** 2 + size_j[1] ** 2)
                                  ) / 2 + 2.0
                    constraints.append(center_dist - min_allowed)

            return np.array(constraints)

        constraints = {'type': 'ineq', 'fun': constraint_func}

        # Method 1: COBYLA
        """result = minimize(
            lambda x: objective_combined(x, building_sizes, park_center, park_radius, road_info, IDEAL_PARK_GAP),
            x0,
            method='COBYLA',
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 1000, 'rhobeg': 1.0}
        )"""

        # Method 2: SLSQP
        result = minimize(
            lambda x: objective_combined(x, building_sizes, park_center, park_radius, road_info, IDEAL_PARK_GAP),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 1000, 'ftol': 1e-8}
        )

        # Method 3: trust-constr
        # result = minimize(
        #     lambda x: objective_combined(x, building_sizes, park_center, park_radius, road_info, IDEAL_PARK_GAP),
        #     x0,
        #     method='trust-constr',
        #     bounds=bounds,
        #     constraints=constraints,
        #     options={'verbose': 1, 'maxiter': 500, 'gtol': 1e-6}
        # )

        if result.success:
            for i, obj in enumerate(buildings):
                obj.location.x, obj.location.y = result.x[2 * i], result.x[2 * i + 1]
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            self.report({'INFO'}, "Optimization successful!")
        else:
            self.report({'WARNING'}, f"Optimization failed: {result.message}")
        return {'FINISHED'}


classes = (CITY_OPTIMIZER_PT_Panel, CITY_OPTIMIZER_OT_Optimize)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
