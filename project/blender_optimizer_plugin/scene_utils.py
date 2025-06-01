import json
import bpy
import math
import numpy as np


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
