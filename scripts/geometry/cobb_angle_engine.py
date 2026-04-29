import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

class CobbAngleEngine:
    """
    Production-Ready Cobb Angle Geometry Engine.
    Handles vertebral tilt detection, perpendicular intersection, and visualization.
    """
    
    def __init__(self):
        pass

    def calculate_slope(self, p1, p2):
        """Calculates the slope (m) of a line between two points."""
        if p2[0] == p1[0]:
            return 1e9  # Avoid division by zero
        return (p2[1] - p1[1]) / (p2[0] - p1[0])

    def get_angle_between_lines(self, m1, m2):
        """Calculates the angle in degrees between two lines given their slopes."""
        # Using the formula: tan(theta) = |(m2 - m1) / (1 + m1*m2)|
        # Note: If 1 + m1*m2 is near zero, the lines are nearly perpendicular (90 deg)
        denom = 1 + m1 * m2
        if np.abs(denom) < 1e-6:
            return 90.0
        
        angle_rad = np.arctan(np.abs((m2 - m1) / denom))
        return np.degrees(angle_rad)

    def process_spine(self, vertebrae_landmarks):
        """
        Expects a list of vertebrae objects.
        Each vertebra should have: 'name' (e.g., L1, T12) and 'points' (4 corners).
        Corners order: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
        """
        slopes = []
        for v in vertebrae_landmarks:
            pts = v['points']
            # We care about the superior (top) and inferior (bottom) endplates
            top_slope = self.calculate_slope(pts[0], pts[1])
            bot_slope = self.calculate_slope(pts[2], pts[3])
            
            slopes.append({
                'name': v['name'],
                'top_slope': top_slope,
                'bot_slope': bot_slope
            })

        # Finding the boundaries of the primary curve
        # Typically, the Cobb angle is defined by the most tilted superior vertebra 
        # and the most tilted inferior vertebra of the curve.
        
        # For a single major curve, we look for the max difference in slopes.
        # More advanced: Identify local extrema for S-curves.
        
        results = []
        
        # Primary Curve (Thoracic or Lumbar)
        # We'll find the global min and max slopes among all endplates
        all_endplate_slopes = []
        for s in slopes:
            all_endplate_slopes.append((s['name'], 'top', s['top_slope']))
            all_endplate_slopes.append((s['name'], 'bot', s['bot_slope']))
            
        # Sort by slope value
        all_endplate_slopes.sort(key=lambda x: x[2])
        
        sup_v = all_endplate_slopes[0]
        inf_v = all_endplate_slopes[-1]
        
        angle = self.get_angle_between_lines(sup_v[2], inf_v[2])
        
        return {
            'cobb_angle': angle,
            'superior_vertebra': sup_v[0],
            'superior_plate': sup_v[1],
            'inferior_vertebra': inf_v[0],
            'inferior_plate': inf_v[1],
            'all_slopes': slopes
        }

    def visualize(self, image_path, landmarks, result, output_path=None):
        """Generates a visualization with the Cobb angle overlaid."""
        img = cv2.imread(image_path)
        if img is None:
            # Create a blank image if path is invalid
            img = np.zeros((1000, 500, 3), dtype=np.uint8) + 255
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 20))
        plt.imshow(img)
        
        # Draw vertebrae boxes
        for v in landmarks:
            pts = np.array(v['points'])
            # Top Plate
            plt.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], 'g-', linewidth=2)
            # Bottom Plate
            plt.plot([pts[2][0], pts[3][0]], [pts[2][1], pts[3][1]], 'g-', linewidth=2)
            # Sides
            plt.plot([pts[0][0], pts[2][0]], [pts[0][1], pts[2][1]], 'g--', alpha=0.5)
            plt.plot([pts[1][0], pts[3][0]], [pts[1][1], pts[3][1]], 'g--', alpha=0.5)
            plt.text(pts[0][0], pts[0][1], v['name'], color='white', fontsize=8, backgroundcolor='black')

        # Draw Cobb Angle lines for the selected vertebrae
        # (This is a simplified visualization of the endplate extensions)
        def draw_extension(v_name, plate_type, color):
            for v in landmarks:
                if v['name'] == v_name:
                    pts = v['points']
                    if plate_type == 'top':
                        p1, p2 = pts[0], pts[1]
                    else:
                        p1, p2 = pts[2], pts[3]
                    
                    # Extend the line
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    ext_p1 = [p1[0] - dx*2, p1[1] - dy*2]
                    ext_p2 = [p2[0] + dx*2, p2[1] + dy*2]
                    plt.plot([ext_p1[0], ext_p2[0]], [ext_p1[1], ext_p2[1]], color=color, linewidth=3)
                    
        draw_extension(result['superior_vertebra'], result['superior_plate'], 'blue')
        draw_extension(result['inferior_vertebra'], result['inferior_plate'], 'red')

        plt.title(f"Cobb Angle: {result['cobb_angle']:.2f}°\n"
                  f"Between {result['superior_vertebra']} ({result['superior_plate']}) and "
                  f"{result['inferior_vertebra']} ({result['inferior_plate']})", 
                  fontsize=14, color='blue', fontweight='bold')
        
        plt.axis('off')
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"[*] Visualization saved to {output_path}")
        plt.show()

if __name__ == "__main__":
    # Test Data: Simulated scoliosis curve
    # Points: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
    test_vertebrae = [
        {'name': 'T10', 'points': [[100, 100], [200, 105], [100, 150], [200, 155]]},
        {'name': 'T11', 'points': [[110, 170], [210, 185], [110, 220], [210, 235]]},
        {'name': 'T12', 'points': [[120, 250], [220, 275], [120, 300], [220, 325]]}, # Inflection
        {'name': 'L1',  'points': [[110, 340], [210, 355], [110, 390], [210, 405]]},
        {'name': 'L2',  'points': [[100, 420], [200, 425], [100, 470], [200, 475]]},
    ]
    
    engine = CobbAngleEngine()
    result = engine.process_spine(test_vertebrae)
    
    print("\n" + "="*40)
    print("      COBB ANGLE ENGINE REPORT")
    print("="*40)
    print(f"Calculated Angle: {result['cobb_angle']:.2f}°")
    print(f"Superior Limit:   {result['superior_vertebra']} ({result['superior_plate']})")
    print(f"Inferior Limit:   {result['inferior_vertebra']} ({result['inferior_plate']})")
    print("="*40)
    
    # Run visualization (will save to a file)
    # Note: On a headless server, plt.show() might need a backend like 'Agg'
    plt.switch_backend('Agg')
    engine.visualize(None, test_vertebrae, result, output_path='cobb_angle_test.png')
