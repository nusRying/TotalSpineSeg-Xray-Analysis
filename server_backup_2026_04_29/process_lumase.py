import os
import glob
from pathlib import Path
from generate_drr import process_case

def process_lumase():
    input_dir = Path("/workspace/data/raw_external/LumASe_Final/L1-L5FineSegMix-663case")
    output_dir = Path("/workspace/data/processed/xray_synthetic_lumase")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files (not containing 'seg')
    all_files = glob.glob(str(input_dir / "*.nii.gz"))
    image_files = [f for f in all_files if "_seg" not in f]
    
    print(f"Found {len(image_files)} individual vertebrae images.")
    
    # Test on the first 5 images
    success_count = 0
    for img_path_str in image_files[:5]:
        img_path = Path(img_path_str)
        # e.g. 101021763532_L4.nii.gz -> mask is 101021763532_L4_seg.nii.gz
        mask_path_str = img_path_str.replace(".nii.gz", "_seg.nii.gz")
        mask_path = Path(mask_path_str)
        
        case_id = img_path.name.replace(".nii.gz", "")
        
        print(f"Processing {case_id}...")
        
        ok = process_case(
            ct_path=img_path,
            label_path=mask_path if mask_path.exists() else None,
            output_dir=output_dir,
            axis=1, # Lateral projection
            case_id=case_id
        )
        if ok:
            success_count += 1
            
    print(f"Successfully processed {success_count} out of 5 test cases.")

if __name__ == "__main__":
    process_lumase()
