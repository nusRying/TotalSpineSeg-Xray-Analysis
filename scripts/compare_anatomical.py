import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def compare():
    img_path = Path("data/xray/images/0001035.png")
    std_path = Path("scratch/compare_masks/standard/0001035.png")
    ana_path = Path("scratch/compare_masks/anatomical/0001035.png")
    
    img = Image.open(img_path).convert("RGB")
    std = Image.open(std_path)
    ana = Image.open(ana_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    
    # Simple overlay
    std_ov = np.array(img).copy()
    std_mask = np.array(std) > 0
    std_ov[std_mask] = [255, 0, 0] # Red
    axes[1].imshow(std_ov)
    axes[1].set_title("Standard Mask (Box)")
    
    ana_ov = np.array(img).copy()
    ana_mask = np.array(ana) > 0
    ana_ov[ana_mask] = [0, 255, 0] # Green
    axes[2].imshow(ana_ov)
    axes[2].set_title("Anatomical Mask (Bone)")
    
    plt.tight_layout()
    plt.savefig("scratch/comparison_0001035.png")
    print("Comparison saved to scratch/comparison_0001035.png")

if __name__ == "__main__":
    compare()
