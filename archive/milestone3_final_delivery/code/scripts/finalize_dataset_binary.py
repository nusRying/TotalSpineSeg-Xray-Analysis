import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def binarize_labels(dataset_path):
    dataset_path = Path(dataset_path)
    for d in ["labelsTr", "labelsTs"]:
        label_dir = dataset_path / d
        if not label_dir.exists():
            continue
        files = list(label_dir.glob("*.png"))
        for f in tqdm(files, desc=f"Binarizing {d}"):
            img = Image.open(f)
            data = np.array(img)
            binary_data = (data > 0).astype(np.uint8)
            Image.fromarray(binary_data).save(f)

if __name__ == "__main__":
    binarize_labels("data/nnUNet/raw/Dataset202_TotalSpineSeg_XRay_FullSpine_AP_Lat")
