import os
import sys
import numpy as np
from PIL import Image
import json
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totalspineseg.xray.geometry.clinical_engine import ClinicalGeometryEngine

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic report from spine mask.")
    parser.add_argument("--mask", type=str, required=True, help="Path to segmentation mask.")
    parser.add_argument("--spacing", type=float, default=0.1, help="Pixel spacing in mm/pixel.")
    parser.add_argument("--output", type=str, default="reports/latest_report.json", help="Output JSON path.")
    args = parser.parse_args()

    if not os.path.exists(args.mask):
        print(f"Mask not found: {args.mask}")
        return

    print(f"Processing mask: {args.mask}")
    mask = np.array(Image.open(args.mask))
    
    # Load Label Map
    label_map_path = "totalspineseg/resources/labels_maps/tss_map.json"
    with open(label_map_path, "r") as f:
        tss_map = json.load(f)
    id_to_name = {v: k for k, v in tss_map.items()}
    
    # Run Engine
    engine = ClinicalGeometryEngine(mask, id_to_name, pixel_spacing=args.spacing)
    report = engine.generate_report()
    
    # Save Report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=4)
    
    print(f"\n=== Diagnostic Summary ===")
    print(f"Status: {report['summary']['status']}")
    print(f"Cervical Lordosis: {report['summary']['cervical_lordosis']:.2f}°")
    print(f"Max Cobb Angle: {report['summary']['max_cobb']:.2f}°")
    print(f"Vertebrae Detected: {report['metadata']['vertebrae_count']}")
    print(f"\nFull report saved to: {args.output}")

if __name__ == "__main__":
    main()
