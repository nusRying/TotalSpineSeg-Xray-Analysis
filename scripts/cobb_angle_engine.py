import numpy as np

def calculate_slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def get_angle_between_slopes(m1, m2):
    # Angle between two lines with slopes m1 and m2
    angle_rad = np.abs(np.arctan((m2 - m1) / (1 + m1 * m2)))
    return np.degrees(angle_rad)

def cobb_angle(vertebrae_landmarks):
    """
    Expects a list of vertebrae, each with 4 points:
    [top_left, top_right, bottom_left, bottom_right]
    """
    slopes = []
    for v in vertebrae_landmarks:
        # Calculate slope of the top surface
        top_slope = calculate_slope(v[0], v[1])
        slopes.append(top_slope)
    
    # Find the two vertebrae with the maximum difference in slope (most tilted)
    m1 = min(slopes)
    m2 = max(slopes)
    
    angle = get_angle_between_slopes(m1, m2)
    return angle

# Example usage:
# v1 = [[10, 10], [50, 12], [10, 40], [50, 42]]
# v2 = [[12, 100], [52, 140], [12, 130], [52, 170]]
# print(f"Cobb Angle: {cobb_angle([v1, v2])}")
