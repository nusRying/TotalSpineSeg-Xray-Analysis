import numpy as np

def vertebral_height_ratio(v_points):
    """
    Points: [Ant-Sup, Post-Sup, Ant-Inf, Post-Inf]
    """
    # Anterior Height (Distance between Ant-Sup and Ant-Inf)
    ah = np.sqrt((v_points[0][0] - v_points[2][0])**2 + (v_points[0][1] - v_points[2][1])**2)
    
    # Posterior Height (Distance between Post-Sup and Post-Inf)
    ph = np.sqrt((v_points[1][0] - v_points[3][0])**2 + (v_points[1][1] - v_points[3][1])**2)
    
    ratio = ah / ph
    
    status = "Normal"
    if ratio < 0.8:
        status = "Wedge Fracture Suspected"
        
    return ah, ph, ratio, status
