import os
import sys
import numpy as np
from PIL import Image
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totalspineseg.xray.geometry.clinical_engine import ClinicalGeometryEngine

def main():
    mask_path = "data/xray/csxa/masks/0001035.png"
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        return

    print(f"Loading mask: {mask_path}")
    mask = np.array(Image.open(mask_path))
    
    # Load Label Map
    label_map_path = "totalspineseg/resources/labels_maps/tss_map.json"
    with open(label_map_path, "r") as f:
        tss_map = json.load(f)
    
    # Invert for Engine (ID -> Name)
    id_to_name = {v: k for k, v in tss_map.items()}
    
    # Initialize Engine
    engine = ClinicalGeometryEngine(mask, id_to_name, pixel_spacing=0.2) # Assuming 0.2mm for test
    
    print("Calculating metrics...")
    metrics = engine.calculate_all_metrics()
    
    # Display some results
    print("\n=== Regional Alignment ===")
    for k in ["cervical_lordosis_deg", "thoracic_kyphosis_deg", "lumbar_lordosis_deg"]:
        val = metrics.get(k)
        if val is not None:
            print(f"{k}: {val:.2f}°")
        
    print("\n=== Sample Segmental Metrics ===")
    # Look for C5-C6 metrics
    c5_c6_ant = "C5_C6_disc_ht_ant_mm"
    if c5_c6_ant in metrics:
        print(f"C5-C6 Disc Height (Ant): {metrics[c5_c6_ant]:.2f} mm")
        print(f"C5-C6 Segmental Angle: {metrics.get('C5_C6_segmental_angle_deg', 0):.2f}°")
        print(f"C5-C6 Listhesis: {metrics.get('C5_C6_neutral_listhesis_mm', 0):.2f} mm")
    
    print("\n=== Height Loss (Fracture Risk) ===")
    for v in ["C2", "C3", "C4", "C5", "C6", "C7"]:
        k = f"{v}_vert_ht_loss_pct"
        if k in metrics:
            print(f"{v} Height Loss: {metrics[k]:.2f}%")

if __name__ == "__main__":
    main()
