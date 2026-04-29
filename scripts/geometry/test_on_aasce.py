import sys
import os
import matplotlib.pyplot as plt

# Add current directory to path so we can import the engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cobb_angle_engine import CobbAngleEngine

def test_aasce_demo():
    """
    Demonstrates the Cobb Angle engine using sample landmarks 
    representative of the AASCE (Scoliosis) dataset.
    """
    print("[*] Running AASCE Cobb Angle Demonstration...")
    
    # Representative landmarks for an S-curve (Thoracolumbar)
    # Each vertebra: [TL, TR, BL, BR]
    landmarks = [
        {'name': 'T7', 'points': [[100, 100], [180, 102], [100, 130], [180, 132]]},
        {'name': 'T8', 'points': [[95, 140], [175, 145], [95, 170], [175, 175]]},
        {'name': 'T9', 'points': [[85, 185], [165, 195], [85, 215], [165, 225]]}, # Tilt Increasing
        {'name': 'T10', 'points': [[75, 235], [155, 255], [75, 265], [155, 285]]}, # Tilt Increasing
        {'name': 'T11', 'points': [[70, 295], [150, 315], [70, 325], [150, 345]]}, # Apex area
        {'name': 'T12', 'points': [[75, 355], [155, 370], [75, 385], [155, 400]]}, # Tilt Decreasing
        {'name': 'L1', 'points': [[90, 410], [170, 415], [90, 440], [170, 445]]}, # Inflection
        {'name': 'L2', 'points': [[110, 455], [190, 450], [110, 485], [190, 480]]}, # Tilting other way
        {'name': 'L3', 'points': [[130, 500], [210, 490], [130, 530], [210, 520]]},
    ]
    
    engine = CobbAngleEngine()
    result = engine.process_spine(landmarks)
    
    print("\n" + "!"*40)
    print("      AASCE SCOLIOSIS CALCULATION")
    print("!"*40)
    print(f"Detected Curve Angle: {result['cobb_angle']:.2f}°")
    print(f"Top boundary:         {result['superior_vertebra']} ({result['superior_plate']})")
    print(f"Bottom boundary:      {result['inferior_vertebra']} ({result['inferior_plate']})")
    print("!"*40)
    
    # Save visualization
    plt.switch_backend('Agg')
    engine.visualize(None, landmarks, result, output_path='aasce_demo_cobb.png')
    print("[+] Demonstration complete. Check 'aasce_demo_cobb.png'.")

if __name__ == "__main__":
    test_aasce_demo()
