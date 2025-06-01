import bpy
from scipy.optimize import minimize
import numpy as np
import math
import json

try:
    import torch
    import torch.nn.functional as F

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

bl_info = {
    "name": "Differentiable City Layout Optimizer",
    "author": "Jurij Nabergoj",
    "version": (3, 0),
    "blender": (3, 0, 0),
    "description": "Optimizes building positions using differentiable programming with PyTorch",
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


def point_to_line_distance_diff(point, line_start, line_end):
    """Differentiable version of point to line distance calculation"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len_sq = torch.dot(line_vec, line_vec) + 1e-8  # Add epsilon

    # Clamp t to [0, 1] range using sigmoid-like function
    t = torch.dot(point_vec, line_vec) / line_len_sq
    t = torch.clamp(t, 0.0, 1.0)  # Clamp to avoid NaN

    projection = line_start + t * line_vec
    distance = torch.norm(point - projection + 1e-8)  # Add epsilon
    return distance


def smooth_max(x, y, alpha=10.0):
    """Smooth differentiable approximation of max(x, y)"""
    exp_x = torch.exp(alpha * x)
    exp_y = torch.exp(alpha * y)
    return (x * exp_x + y * exp_y) / (exp_x + exp_y)


def smooth_min(x, y, alpha=10.0):
    """Smooth differentiable approximation of min(x, y)"""
    return -smooth_max(-x, -y, alpha)


def distance_to_ring_road_diff(point, center, inner_radius, outer_radius):
    """Differentiable distance to ring road"""
    dist_to_center = torch.norm(point - center + 1e-8)  # Add small epsilon

    # Distance to inner edge
    inner_dist = torch.abs(dist_to_center - inner_radius)
    # Distance to outer edge
    outer_dist = torch.abs(dist_to_center - outer_radius)

    # Return minimum distance to either edge
    return torch.min(inner_dist, outer_dist)


def distance_to_segment_road_diff(point, start, end, width):
    """Differentiable distance to road segment"""
    dist_to_line = point_to_line_distance_diff(point, start, end)
    return dist_to_line - width / 2


def building_to_park_distance_diff(building_pos, building_size, park_center, ring_outer_radius):
    """Differentiable distance from building edge to park accessible area"""
    dist_to_center = torch.norm(building_pos - park_center + 1e-8)  # Add epsilon
    building_diagonal = torch.sqrt(building_size[0] ** 2 + building_size[1] ** 2 + 1e-8)
    closest_building_edge_to_center = torch.clamp(dist_to_center - building_diagonal / 2, min=0.0)
    return closest_building_edge_to_center - ring_outer_radius


def objective_function_diff(x, building_sizes, park_center, ring_outer_radius, num_buildings):
    """Differentiable objective function"""
    IDEAL_PARK_GAP = 1.0
    total_squared_error = torch.tensor(0.0)

    for i in range(num_buildings):
        pos_2d = torch.stack([x[2 * i], x[2 * i + 1]])
        size = torch.tensor(building_sizes[i], dtype=torch.float32)

        raw_distance = building_to_park_distance_diff(pos_2d, size, park_center, ring_outer_radius)
        error = raw_distance - IDEAL_PARK_GAP
        total_squared_error += error ** 2

    return total_squared_error


def combined_objective_diff(x, building_sizes, park_center, ring_outer_radius, num_buildings):
    """Simplified differentiable combined objective"""
    IDEAL_PARK_GAP = 1.0

    total_distance_error = torch.tensor(0.0, dtype=torch.float32)

    positions = x.reshape(num_buildings, 2)

    for i in range(num_buildings):
        pos_i = positions[i]
        size_i = torch.tensor(building_sizes[i], dtype=torch.float32)

        # Distance to park edge
        raw_distance = building_to_park_distance_diff(pos_i, size_i, park_center, ring_outer_radius)
        error = raw_distance - IDEAL_PARK_GAP
        total_distance_error += error ** 2

        # Building separation penalty
        for j in range(i + 1, num_buildings):
            pos_j = positions[j]
            dist = torch.norm(pos_i - pos_j + 1e-8)  # Add epsilon
            # Penalty when too close (less than 2 units)
            if dist < 2.0:
                total_distance_error += (2.0 - dist) ** 2

    return total_distance_error


def constraint_violations_diff(x, building_sizes, road_segments, ring_roads, num_buildings):
    """Simplified differentiable constraint violations"""
    violations = []
    positions = x.reshape(num_buildings, 2)

    for i in range(num_buildings):
        pos_i = positions[i]

        # Check building center distance to roads
        min_road_distance = torch.tensor(1000.0, dtype=torch.float32)

        # Check distance to road segments
        for road_start, road_end, road_width in road_segments:
            dist = distance_to_segment_road_diff(pos_i, road_start, road_end, road_width)
            min_road_distance = torch.min(min_road_distance, dist)

        # Check distance to ring roads
        for ring_center, inner_r, outer_r in ring_roads:
            dist = distance_to_ring_road_diff(pos_i, ring_center, inner_r, outer_r)
            min_road_distance = torch.min(min_road_distance, dist)

        # Road clearance constraint
        violations.append(min_road_distance - 2.0)  # Must be >= 2.0 units away

        # Building separation constraints (simplified)
        for j in range(i + 1, num_buildings):
            pos_j = positions[j]
            center_dist = torch.norm(pos_i - pos_j + 1e-8)
            min_allowed = 4.0  # Simplified minimum distance
            violations.append(center_dist - min_allowed)

    return torch.stack(violations) if violations else torch.tensor([], dtype=torch.float32)


class DifferentiableOptimizer:
    """Wrapper class for differentiable optimization with PyTorch"""

    def __init__(self, building_sizes, park_center, road_info):
        self.building_sizes = building_sizes
        self.park_center = torch.tensor(park_center, dtype=torch.float32)
        self.num_buildings = len(building_sizes)

        # Extract ring road info
        ring_road = next((road for road in road_info if road['type'] == 'ring'), None)
        self.ring_outer_radius = torch.tensor(ring_road['outer_radius'] if ring_road else 12.0, dtype=torch.float32)

        # Prepare road data for PyTorch
        self.road_segments = []
        self.ring_roads = []

        for road in road_info:
            if road['type'] == 'segment':
                self.road_segments.append((
                    torch.tensor(road['start'], dtype=torch.float32),
                    torch.tensor(road['end'], dtype=torch.float32),
                    torch.tensor(road['width'], dtype=torch.float32)
                ))
            elif road['type'] == 'ring':
                self.ring_roads.append((
                    torch.tensor(road['center'][:2], dtype=torch.float32),
                    torch.tensor(road['inner_radius'], dtype=torch.float32),
                    torch.tensor(road['outer_radius'], dtype=torch.float32)
                ))

    def _objective_wrapper(self, x):
        """Wrapper for PyTorch computation"""
        return combined_objective_diff(
            x, self.building_sizes, self.park_center,
            self.ring_outer_radius, self.num_buildings
        )

    def _constraint_wrapper(self, x):
        """Wrapper for PyTorch computation"""
        return constraint_violations_diff(
            x, self.building_sizes, self.road_segments,
            self.ring_roads, self.num_buildings
        )

    def objective(self, x):
        """Objective function for scipy"""
        try:
            x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=False)
            obj_val = self._objective_wrapper(x_torch)
            result = float(obj_val.detach().numpy())

            # Check for NaN values, return large (1e6) penalty for errors
            if np.isnan(result):
                print("Warning: NaN detected in objective, returning large value")
                return 1e6

            return result
        except Exception as e:
            print(f"Error in objective: {e}")
            return 1e6

    def objective_grad(self, x):
        """Gradient of objective function using PyTorch autograd"""
        try:
            x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            obj_val = self._objective_wrapper(x_torch)

            # Check for NaN in objective before computing gradients
            if torch.isnan(obj_val):
                print("Warning: NaN in objective before gradient computation")
                return np.zeros_like(x)

            # Compute gradients
            obj_val.backward()
            grad_val = x_torch.grad

            if grad_val is None:
                print("Warning: No gradients computed")
                return np.zeros_like(x)

            result = grad_val.detach().numpy()

            # Check for NaN values in gradients
            if np.any(np.isnan(result)):
                print("Warning: NaN detected in gradients, replacing with zeros")
                result = np.nan_to_num(result, nan=0.0)

            return result
        except Exception as e:
            print(f"Error in gradient computation: {e}")
            return np.zeros_like(x)  # Return zero gradients as fallback

    def constraints(self, x):
        """Constraint function for scipy"""
        try:
            x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=False)
            violations = self._constraint_wrapper(x_torch)
            result = violations.detach().numpy()

            # Check for NaN values and replace with large negative numbers
            if np.any(np.isnan(result)):
                print("Warning: NaN detected in constraints, replacing with -1000")
                result = np.nan_to_num(result, nan=-1000.0)

            return result
        except Exception as e:
            print(f"Error in constraints: {e}")
            # Return feasible constraints as fallback
            return np.ones(len(x)) * 10.0  # All constraints satisfied


class CITY_OPTIMIZER_PT_Panel(bpy.types.Panel):
    bl_idname = "CITY_OPTIMIZER_PT_Panel"
    bl_label = "Differentiable City Layout Optimizer"
    bl_category = "City Optimizer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        layout = self.layout
        if PYTORCH_AVAILABLE:
            layout.operator("object.optimize_city_layout_diff")
        else:
            layout.label(text="PyTorch not available")
            layout.label(text="Install: pip install torch")


class CITY_OPTIMIZER_OT_Optimize(bpy.types.Operator):
    bl_idname = "object.optimize_city_layout_diff"
    bl_label = "Optimize Layout"
    bl_description = "Optimize building positions using differentiable programming with PyTorch"

    def execute(self, context):
        if not PYTORCH_AVAILABLE:
            self.report({'ERROR'}, "PyTorch not available. Install with: pip install torch")
            return {'CANCELLED'}

        print("Starting DIFFERENTIABLE city layout optimization with PyTorch...")
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

        # Initialize optimizer
        optimizer = DifferentiableOptimizer(building_sizes, park_center, road_info)

        # Initial positions
        x0 = []
        for building in buildings:
            x0.extend([building.location.x, building.location.y])
            print(f"Starting position for {building.name}: ({building.location.x:.2f}, {building.location.y:.2f})")

        x0 = np.array(x0)

        # Bounds
        max_bound = 30.0
        bounds = [(park_center[i % 2] - max_bound, park_center[i % 2] + max_bound)
                  for i in range(2 * len(buildings))]

        print("Starting optimization with PyTorch gradients...")

        # Test objective and constraints at initial point
        print("Testing initial point...")
        try:
            initial_obj = optimizer.objective(x0)
            initial_constraints = optimizer.constraints(x0)
            print(f"Initial objective: {initial_obj}")
            print(
                f"Initial constraints (first 5): {initial_constraints[:5] if len(initial_constraints) > 0 else'No constraints'}")

            if np.isnan(initial_obj):
                self.report({'ERROR'}, "Initial objective is NaN - check your scene setup")
                return {'CANCELLED'}

            if len(initial_constraints) > 0 and np.all(initial_constraints < -100):
                self.report({'ERROR'}, "All constraints heavily violated - check building/road positions")
                return {'CANCELLED'}

        except Exception as e:
            self.report({'ERROR'}, f"Error evaluating initial point: {e}")
            return {'CANCELLED'}

        # Try unconstrained optimization
        print("Starting with unconstrained optimization first...")

        # First run unconstrained to get a feasible starting point
        result_unconstrained = minimize(
            optimizer.objective,
            x0,
            method='BFGS',  # Use BFGS instead of SLSQP for unconstrained
            jac=optimizer.objective_grad,
            bounds=bounds,
            options={
                'disp': True,
                'maxiter': 100,
                'gtol': 1e-4
            }
        )

        print(f"Unconstrained result: {result_unconstrained.success}, final value: {result_unconstrained.fun}")

        if result_unconstrained.success and result_unconstrained.fun < 1e5:
            # Use the unconstrained result as starting point for constrained optimization
            print("Now running constrained optimization...")
            x_start = result_unconstrained.x
        else:
            print("Unconstrained optimization failed, using original starting point")
            x_start = x0

        # Now try constrained optimization with the better starting point
        if len(initial_constraints) > 0:
            constraints = {'type': 'ineq', 'fun': optimizer.constraints}
        else:
            print("Warning: No constraints defined, skipping constrained phase")
            constraints = None

        if constraints is not None:
            result = minimize(
                optimizer.objective,
                x_start,
                method='SLSQP',
                jac=optimizer.objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={
                    'disp': True,
                    'maxiter': 200,
                    'ftol': 1e-5,
                    'eps': 1e-5
                }
            )
        else:
            result = result_unconstrained

        print(f"Optimization completed. Success: {result.success}")
        print(f"Final objective value: {result.fun:.6f}")
        print(f"Message: {result.message}")

        if result.success or result.fun < 1e-6:
            print("Updating building positions...")
            for i, obj in enumerate(buildings):
                obj.location.x, obj.location.y = result.x[2 * i], result.x[2 * i + 1]
                print(f"New position for {obj.name}: ({obj.location.x:.2f}, {obj.location.y:.2f})")

            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

            self.report({'INFO'}, f"PyTorch differentiable optimization successful! Final cost: {result.fun:.6f}")
        else:
            self.report({'WARNING'}, f"Optimization failed: {result.message}")

        return {'FINISHED'}


classes = (CITY_OPTIMIZER_PT_Panel, CITY_OPTIMIZER_OT_Optimize)


def register():
    [bpy.utils.register_class(cls) for cls in classes]


def unregister():
    [bpy.utils.unregister_class(cls) for cls in classes]


if __name__ == "__main__":
    register()
