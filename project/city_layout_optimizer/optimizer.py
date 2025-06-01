import numpy as np
import math
from .geometry_utils import point_to_line_distance, buildings_overlap


def distance_building_to_park_edge(building_pos, building_size, park_center, park_radius, road_info):
    """Calculate distance from building edge to park accessible area (road edge)"""
    ring_road = next((road for road in road_info if road['type'] == 'ring'), None)
    target_radius = ring_road['outer_radius'] if ring_road else park_radius + 6.0
    dist_to_center = np.linalg.norm(np.array(building_pos) - park_center)
    building_diagonal = math.sqrt(building_size[0] ** 2 + building_size[1] ** 2)
    closest_building_edge_to_center = max(0, dist_to_center - building_diagonal / 2)
    return closest_building_edge_to_center - target_radius


def objective_combined(x, building_sizes, park_center, park_radius, road_info, IDEAL_PARK_GAP):
    weight_distance = 1.0
    weight_sunlight = 0.3
    weight_spread = 0.2

    num_buildings = len(building_sizes)
    total_distance_error = 0.0
    sunlight_penalty = 0.0
    spread_penalty = 0.0

    building_positions = [
        np.array([x[2 * i], x[2 * i + 1]]) for i in range(num_buildings)
    ]

    for i in range(num_buildings):
        pos_i = building_positions[i]
        size_i = building_sizes[i]

        # (1) Park distance objective
        dist_to_edge = distance_building_to_park_edge(pos_i, size_i, park_center, park_radius, road_info)
        error = dist_to_edge - IDEAL_PARK_GAP
        total_distance_error += error ** 2

        # (2) Sunlight exposure penalty
        for j, pos_j in enumerate(building_positions):
            if i != j and pos_j[1] < pos_i[1]:
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < 10:
                    sunlight_penalty += (10 - dist) ** 2

        # (3) Spread penalty
        for j in range(i + 1, num_buildings):
            pos_j = building_positions[j]
            dist = np.linalg.norm(pos_i - pos_j)
            spread_penalty += 1.0 / (dist + 1e-3)

    return (
            weight_distance * total_distance_error +
            weight_sunlight * sunlight_penalty +
            weight_spread * spread_penalty
    )
