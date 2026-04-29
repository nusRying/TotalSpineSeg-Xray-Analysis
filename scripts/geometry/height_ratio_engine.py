import numpy as np
import matplotlib.pyplot as plt
import cv2

class HeightRatioEngine:
    """
    Vertebral Height Ratio (VHR) Engine.
    Used for lateral X-ray analysis to detect wedge fractures (Compression Fractures).
    """
    
    def __init__(self, fracture_threshold=0.8):
        self.fracture_threshold = fracture_threshold

    def calculate_distance(self, p1, p2):
        """Calculates Euclidean distance between two points."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def process_vertebra(self, points):
        """
        Calculates heights and ratio for a single vertebra.
        Points order: [Ant-Sup, Post-Sup, Ant-Inf, Post-Inf]
        """
        # Anterior Height (Distance between Ant-Sup and Ant-Inf)
        ah = self.calculate_distance(points[0], points[2])
        
        # Posterior Height (Distance between Post-Sup and Post-Inf)
        ph = self.calculate_distance(points[1], points[3])
        
        # Middle Height (Optional, but often used: Avg of Ant and Post or center measurement)
        # For simplicity, we stick to Ant/Post ratio
        
        ratio = ah / ph if ph != 0 else 1.0
        
        status = "Normal"
        if ratio < self.fracture_threshold:
            status = "Wedge Fracture Suspected"
        elif ratio > 1.2:
            status = "Biconcave/Anomalous Shape"
            
        return {
            'anterior_height': ah,
            'posterior_height': ph,
            'ratio': ratio,
            'status': status
        }

    def process_spine(self, vertebrae_data):
        """
        Processes a list of vertebrae.
        vertebrae_data: list of dicts with 'name' and 'points'.
        """
        results = []
        for v in vertebrae_data:
            analysis = self.process_vertebra(v['points'])
            analysis['name'] = v['name']
            results.append(analysis)
        return results

    def visualize(self, image_path, vertebrae_data, results, output_path=None):
        """Generates visualization for lateral X-ray height ratios."""
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((1000, 500, 3), dtype=np.uint8) + 240
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 18))
        plt.imshow(img)
        
        for i, v in enumerate(vertebrae_data):
            pts = np.array(v['points'])
            res = results[i]
            
            # Color based on status
            color = 'blue' if res['status'] == "Normal" else 'red'
            
            # Draw Vertebra Body
            # Ant-Sup to Post-Sup
            plt.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=color, linewidth=2)
            # Ant-Inf to Post-Inf
            plt.plot([pts[2][0], pts[3][0]], [pts[2][1], pts[3][1]], color=color, linewidth=2)
            # Anterior Height line
            plt.plot([pts[0][0], pts[2][0]], [pts[0][1], pts[2][1]], 'g--', linewidth=2, label='Anterior' if i==0 else "")
            # Posterior Height line
            plt.plot([pts[1][0], pts[3][0]], [pts[1][1], pts[3][1]], 'k--', linewidth=2, label='Posterior' if i==0 else "")
            
            # Labels
            mid_y = (pts[0][1] + pts[2][1]) / 2
            plt.text(pts[0][0] - 40, mid_y, f"R:{res['ratio']:.2f}", color=color, fontweight='bold', fontsize=9)
            plt.text(pts[1][0] + 10, pts[1][1], v['name'], color='white', backgroundcolor='black', fontsize=8)

        plt.title("Vertebral Height Ratio Analysis (Lateral View)", fontsize=16)
        plt.legend()
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"[*] Height Ratio visualization saved to {output_path}")
        plt.show()

if __name__ == "__main__":
    # Test Data: Simulated lateral lumbar spine with one wedge fracture
    test_lumbar = [
        {'name': 'L1', 'points': [[150, 100], [250, 100], [150, 180], [250, 180]]}, # Normal (1.0)
        {'name': 'L2', 'points': [[150, 200], [250, 200], [165, 260], [250, 280]]}, # Wedge Fracture (AH < PH)
        {'name': 'L3', 'points': [[150, 300], [250, 300], [150, 380], [250, 380]]}, # Normal
    ]
    
    engine = HeightRatioEngine(fracture_threshold=0.85)
    results = engine.process_spine(test_lumbar)
    
    print("\n" + "="*40)
    print("      HEIGHT RATIO ENGINE REPORT")
    print("="*40)
    for r in results:
        print(f"Vertebra {r['name']}: Ratio {r['ratio']:.2f} -> {r['status']}")
    print("="*40)
    
    plt.switch_backend('Agg')
    engine.visualize(None, test_lumbar, results, output_path='height_ratio_test.png')
