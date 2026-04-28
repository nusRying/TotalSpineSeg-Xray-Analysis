"""
TotalSpineSeg X-Ray: Clinical Geometry Engine
Version: 2026.04.28

This module transforms anatomical segmentations into clinical metrics:
1. Cobb's Angle Calculation (Scoliosis Analytics)
2. Vertebral Height Ratios (Fracture Detection)
3. Surgical Coordinate Export (JSON Mesh landmarks)
"""

import numpy as np
from scipy.spatial import ConvexHull

def calculate_vertebral_slope(component_mask):
    """
    Calculates the orientation (slope) of a vertebra body.
    We use the principle axis of the component.
    """
    y, x = np.where(component_mask > 0)
    if len(x) < 5:
        return 0
    
    # Simple linear regression on the coordinates to find the tilt
    coeffs = np.polyfit(x, y, 1)
    angle_rad = np.arctan(coeffs[0])
    return np.degrees(angle_rad)

def calculate_cobb_angle(component_masks):
    """
    Finds the maximum Cobb's angle in a sequence of vertebrae.
    Returns (angle, superior_index, inferior_index)
    """
    slopes = [calculate_vertebral_slope(m) for m in component_masks]
    
    max_cobb = 0
    indices = (0, 0)
    
    for i in range(len(slopes)):
        for j in range(i + 1, len(slopes)):
            diff = abs(slopes[i] - slopes[j])
            if diff > max_cobb:
                max_cobb = diff
                indices = (i, j)
                
    return max_cobb, indices

def calculate_height_ratio(component_mask):
    """
    Calculates Anterior-to-Posterior (A/P) height ratio.
    Used for detecting Compression Fractures (VCF).
    """
    y, x = np.where(component_mask > 0)
    if len(x) < 10:
        return 1.0
    
    # Identify Anterior (left-most in standard LAT) vs Posterior (right-most)
    # We sample the heights at the 10th and 90th percentile of X
    x_min, x_max = np.min(x), np.max(x)
    ant_x = int(x_min + 0.1 * (x_max - x_min))
    post_x = int(x_min + 0.9 * (x_max - x_min))
    
    ant_y = y[x == ant_x]
    post_y = y[x == post_x]
    
    if len(ant_y) == 0 or len(post_y) == 0:
        return 1.0
        
    ant_height = np.max(ant_y) - np.min(ant_y)
    post_height = np.max(post_y) - np.min(post_y)
    
    if post_height == 0:
        return 1.0
        
    return ant_height / post_height

def get_surgical_landmarks(component_mask):
    """
    Extracts the 4 corners and centroid for surgical JSON export.
    """
    y, x = np.where(component_mask > 0)
    points = np.column_stack((x, y))
    
    if len(points) < 3:
        return {}
        
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]
    
    # Find TL, TR, BR, BL from the hull
    tl = hull_pts[np.argmin(hull_pts[:, 0] + hull_pts[:, 1])]
    br = hull_pts[np.argmax(hull_pts[:, 0] + hull_pts[:, 1])]
    tr = hull_pts[np.argmax(hull_pts[:, 0] - hull_pts[:, 1])]
    bl = hull_pts[np.argmin(hull_pts[:, 0] - hull_pts[:, 1])]
    
    centroid = np.mean(points, axis=0)
    
    return {
        "corners": {
            "top_left": tl.tolist(),
            "top_right": tr.tolist(),
            "bottom_right": br.tolist(),
            "bottom_left": bl.tolist()
        },
        "centroid": centroid.tolist(),
        "height_ratio": calculate_height_ratio(component_mask)
    }
