import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import json
import sys

# Add project root to path so we can import totalspineseg
sys.path.append(str(Path(__file__).resolve().parents[1]))

from totalspineseg.xray.inference import stage_single_image
from totalspineseg.xray.postprocess import postprocess_folder

def test_scaling_logic():
    print("🚀 Starting Milestone 4 Validation: Adaptive Scaling...")
    
    # 1. Setup temporary workspace
    test_dir = Path("tmp_milestone4_test")
    input_dir = test_dir / "input_clinical"
    output_dir = test_dir / "output_processed"
    staged_dir = test_dir / "staged_ai"
    raw_pred_dir = test_dir / "raw_predictions"
    
    for d in [input_dir, output_dir, staged_dir, raw_pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Create a high-resolution "Clinical" image (2048x2048)
    orig_size = (2048, 2048)
    img_path = input_dir / "test_case_0000.png"
    Image.fromarray(np.zeros(orig_size, dtype=np.uint8)).save(img_path)
    print(f"✅ Created dummy clinical image: {orig_size}")

    # 3. Test Staging (Downscaling to 1024px)
    target_size = 1024
    staged_path = staged_dir / "test_case_0000.png"
    shape = stage_single_image(img_path, staged_path, target_size=target_size)
    
    with Image.open(staged_path) as staged_img:
        staged_size = staged_img.size
        
    print(f"✅ Staged image size: {staged_size} (Expected: 1024x1024)")
    assert staged_size == (1024, 1024), "Staging failed to resize correctly!"
    assert shape == orig_size, "Staging failed to return original shape!"

    # 4. Create a dummy "AI Prediction" at 1024px
    # We simulate a "blob" at the center
    pred_mask = np.zeros((1024, 1024), dtype=np.uint8)
    pred_mask[500:600, 500:600] = 1 
    pred_path = raw_pred_dir / "test_case.png"
    Image.fromarray(pred_mask).save(pred_path)
    print(f"✅ Created dummy AI prediction: {pred_mask.shape}")

    # 5. Test Post-processing (Upscaling back to 2048px)
    staged_shapes = {"test_case": orig_size}
    
    postprocess_folder(
        input_dir=raw_pred_dir,
        output_dir=output_dir,
        staged_shapes=staged_shapes,
        prediction_mode="binary",
        overwrite=True
    )
    
    # 6. Verify final output
    final_mask_path = output_dir / "binary" / "test_case.png"
    with Image.open(final_mask_path) as final_mask:
        final_size = final_mask.size
        
    print(f"✅ Final output mask size: {final_size} (Expected: 2048x2048)")
    assert final_size == orig_size, "Post-processing failed to rescale back to original size!"

    print("\n✨ MILESTONE 4 VALIDATION SUCCESSFUL ✨")
    print("The adaptive scaling logic correctly standardizes input and restores output resolution.")
    
    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    try:
        test_scaling_logic()
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
