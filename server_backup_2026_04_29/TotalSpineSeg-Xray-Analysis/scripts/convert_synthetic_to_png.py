import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

def convert_nii_to_png(src_dir, dst_dir, is_label=False):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(src_dir.glob("*.nii.gz"))
    for f in files:
        data = nib.load(str(f)).get_fdata().T # Transpose to match image orientation
        data = np.flipud(data) # Flip vertically to match typical X-ray orientation
        
        # Determine output filename
        # images have _0000 suffix, labels don't
        stem = f.name.replace(".nii.gz", "")
        out_path = dst_dir / f"{stem}.png"
        
        if is_label:
            # Labels: preserve integer values, convert to uint8 or uint16
            Image.fromarray(data.astype(np.uint8)).save(out_path)
        else:
            # Images: normalize to 0-255
            img_min, img_max = data.min(), data.max()
            if img_max > img_min:
                normalized = (data - img_min) / (img_max - img_min) * 255.0
            else:
                normalized = data * 0
            Image.fromarray(normalized.astype(np.uint8)).save(out_path)
        print(f"Converted {f.name} to {out_path.name}")

if __name__ == "__main__":
    # AP
    convert_nii_to_png("data/processed/phantom_full_ap/train", "data/xray/fullspine/images", is_label=False)
    # The projected labels are in the same folder
    convert_nii_to_png("data/processed/phantom_full_ap/train", "data/xray/fullspine/labels", is_label=True)
    
    # Lateral (Adding 'lat' suffix to avoid ID collision)
    # I need to rename them slightly
    src_lat_img = Path("data/processed/phantom_full_lateral/train")
    dst_img = Path("data/xray/fullspine/images")
    dst_lbl = Path("data/xray/fullspine/labels")
    
    for f in src_lat_img.glob("*_0000.nii.gz"):
        data = nib.load(str(f)).get_fdata().T
        data = np.flipud(data)
        img_min, img_max = data.min(), data.max()
        normalized = (data - img_min) / (img_max - img_min) * 255.0
        new_name = f.name.replace("sub-phantom", "sub-phantom-lat").replace(".nii.gz", ".png")
        Image.fromarray(normalized.astype(np.uint8)).save(dst_img / new_name)
        
        # Corresponding label
        lbl_f = src_lat_img / f.name.replace("_0000.nii.gz", ".nii.gz")
        if lbl_f.exists():
            lbl_data = nib.load(str(lbl_f)).get_fdata().T
            lbl_data = np.flipud(lbl_data)
            lbl_name = lbl_f.name.replace("sub-phantom", "sub-phantom-lat").replace(".nii.gz", ".png")
            Image.fromarray(lbl_data.astype(np.uint8)).save(dst_lbl / lbl_name)
            print(f"Converted Lateral {f.name} to {new_name}")
