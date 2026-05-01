import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totalspineseg.xray.geometry.clinical_engine import ClinicalGeometryEngine, VertebraGeometry

def main():
    landmarks_csv = "data/xray/csxa/csxa_landmarks.csv"
    masks_dir = "data/xray/csxa/masks"
    label_map_path = "totalspineseg/resources/labels_maps/tss_map.json"
    
    if not os.path.exists(landmarks_csv):
        print(f"Landmarks not found: {landmarks_csv}")
        return

    # Load Label Map
    with open(label_map_path, "r") as f:
        tss_map = json.load(f)
    id_to_name = {v: k for k, v in tss_map.items()}
    
    # Map CSV vertebra_order to Label Name
    # 1=C2, 2=C3, 3=C4, 4=C5, 5=C6, 6=C7
    order_to_name = {
        1: "C2", 2: "C3", 3: "C4", 4: "C5", 5: "C6", 6: "C7"
    }

    print("Loading landmarks...")
    df = pd.read_csv(landmarks_csv)
    
    # Filter for cases that have masks
    if not os.path.exists(masks_dir):
        print(f"Masks directory not found: {masks_dir}")
        return
        
    available_cases = [f.split(".")[0] for f in os.listdir(masks_dir) if f.endswith(".png")]
    df = df[df['case_id'].astype(str).str.zfill(7).isin(available_cases)]
    
    case_ids = df['case_id'].unique()
    print(f"Found {len(case_ids)} cases for validation.")

    results = []
    
    for case_id in tqdm(case_ids):
        case_str = str(case_id).zfill(7)
        mask_path = os.path.join(masks_dir, f"{case_str}.png")
        
        if not os.path.exists(mask_path):
            continue
            
        # 1. Calculate GT Metrics from Landmarks
        case_df = df[df['case_id'] == case_id]
        gt_vertebrae = []
        for _, row in case_df.iterrows():
            name = order_to_name.get(row['vertebra_order'])
            if not name: continue
            
            landmarks = {
                "SA": np.array([row['x1'], row['y1']]),
                "SP": np.array([row['x2'], row['y2']]),
                "IP": np.array([row['x3'], row['y3']]),
                "IA": np.array([row['x4'], row['y4']])
            }
            vg = VertebraGeometry(label_name=name, landmarks=landmarks)
            gt_vertebrae.append(vg)
            
        gt_engine = ClinicalGeometryEngine(vertebrae=gt_vertebrae, pixel_spacing=1.0)
        gt_metrics = gt_engine.calculate_all_metrics()
        
        # 2. Calculate Pred Metrics from Mask
        mask = np.array(Image.open(mask_path))
        pred_engine = ClinicalGeometryEngine(mask=mask, label_map=id_to_name, pixel_spacing=1.0)
        pred_metrics = pred_engine.calculate_all_metrics()
        
        # 3. Compare common metrics
        for k in gt_metrics:
            if k in pred_metrics:
                results.append({
                    "case_id": case_str,
                    "metric": k,
                    "gt": gt_metrics[k],
                    "pred": pred_metrics[k],
                    "error": abs(gt_metrics[k] - pred_metrics[k])
                })

    if not results:
        print("No metrics compared.")
        return
        
    res_df = pd.DataFrame(results)
    
    # Summary Statistics
    summary = res_df.groupby('metric')['error'].agg(['mean', 'std', 'max']).reset_index()
    summary.columns = ['Metric', 'MAE', 'StdErr', 'MaxErr']
    
    print("\n=== Geometric Pipeline Validation Summary ===")
    print(summary.to_string(index=False))
    
    # Save detailed results
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/geometric_validation_results.csv"
    res_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()
