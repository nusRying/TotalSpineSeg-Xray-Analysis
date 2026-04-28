import json
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Map CSXA JSON landmarks to canonical CSV schema.")
    parser.add_argument("--json-dir", type=Path, required=True, help="Directory containing CSXA JSON files.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output canonical CSV file.")
    return parser.parse_args()

def main():
    args = parse_args()
    json_files = list(args.json_dir.glob("*.json"))
    
    # Canonical columns: case_id, vertebra_order, x1, y1, x2, y2, x3, y3, x4, y4
    fieldnames = ["case_id", "vertebra_order", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for json_path in tqdm(json_files, desc="Mapping landmarks"):
            case_id = json_path.stem
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Group points by vertebra (e.g., C2, C3, ...)
            vertebrae_data = {}
            for shape in data["shapes"]:
                label = shape["label"]
                # Label format: "C3 top left"
                parts = label.split()
                if len(parts) < 2:
                    continue
                v_name = parts[0] # "C3"
                point_type = " ".join(parts[1:]) # "top left"
                
                if v_name not in vertebrae_data:
                    vertebrae_data[v_name] = {}
                
                # CSXA points are [x, y]
                vertebrae_data[v_name][point_type] = shape["points"][0]
            
            # Map to canonical 4-point polygon (top-left, top-right, bottom-right, bottom-left)
            # Sequence: C2, C3, C4, C5, C6, C7
            v_names = ["C2", "C3", "C4", "C5", "C6", "C7"]
            for i, v_name in enumerate(v_names, start=1):
                if v_name not in vertebrae_data:
                    continue
                
                v = vertebrae_data[v_name]
                
                # We need top-left, top-right, bottom-right, bottom-left (4 corners)
                # Some CSXA vertebrae might be missing corners.
                try:
                    tl = v.get("top left")
                    tr = v.get("top right")
                    br = v.get("bottom right")
                    bl = v.get("bottom left")
                    
                    # For C2, the top might be missing (it often uses "centroid" or just bottom)
                    # If top is missing, we approximate a small box or skip
                    if not tl or not tr:
                        if v_name == "C2" and bl and br:
                            # Heuristic for C2: Create a top by offsetting bottom upwards
                            # Centroid is usually above bottom
                            centroid = v.get("centroid")
                            if centroid:
                                # Simple box around centroid or offset from bottom
                                dy = bl[1] - centroid[1]
                                tl = [bl[0], bl[1] - dy * 1.5]
                                tr = [br[0], br[1] - dy * 1.5]
                            else:
                                # Fallback: small offset
                                tl = [bl[0], bl[1] - 40]
                                tr = [br[0], br[1] - 40]
                        else:
                            continue

                    if not tl or not tr or not br or not bl:
                        continue

                    writer.writerow({
                        "case_id": case_id,
                        "vertebra_order": i, # 1 for C2, 2 for C3, etc.
                        "x1": tl[0], "y1": tl[1],
                        "x2": tr[0], "y2": tr[1],
                        "x3": br[0], "y3": br[1],
                        "x4": bl[0], "y4": bl[1]
                    })
                except Exception:
                    continue

if __name__ == "__main__":
    main()
