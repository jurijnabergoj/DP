import numpy as np
import math


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


def buildings_overlap(pos1, size1, pos2, size2, min_gap=2.0):
    """Check if two buildings overlap with a minimum gap"""
    center_dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
    min_dist = (math.sqrt(size1[0] ** 2 + size1[1] ** 2) +
                math.sqrt(size2[0] ** 2 + size2[1] ** 2)) / 2 + min_gap
    return center_dist < min_dist
