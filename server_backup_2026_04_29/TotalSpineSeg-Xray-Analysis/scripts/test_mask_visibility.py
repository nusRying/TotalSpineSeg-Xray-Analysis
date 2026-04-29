import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from totalspineseg.xray.common import save_label_image

def test_mask_visibility():
    print("🔬 Testing Mask Visibility Fix...")
    
    test_path = Path("test_visibility_mask.png")
    
    # 1. Create a dummy mask with low integer values (1, 2, 3)
    # These would normally look black in a standard viewer
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:90] = 1   # C1 area
    mask[40:60, 10:90] = 2   # C2 area
    mask[70:90, 10:90] = 3   # C3 area
    
    # 2. Save it using our updated function
    save_label_image(mask, test_path)
    print(f"✅ Mask saved to {test_path}")
    
    # 3. Verify the file format
    with Image.open(test_path) as img:
        print(f"📸 Image Mode: {img.mode} (Expected: P)")
        assert img.mode == "P", "Image should be in Palette mode!"
        
        # Check if color at index 1 is not black (0,0,0)
        palette = img.getpalette()
        # Each color is 3 bytes (R,G,B). Index 1 starts at byte 3.
        r1, g1, b1 = palette[3:6]
        print(f"🎨 Color for Label 1: RGB({r1}, {g1}, {b1})")
        assert (r1, g1, b1) != (0, 0, 0), "Label 1 should NOT be black in the palette!"
        
        # Re-load data to ensure integer values are preserved
        data = np.asarray(img)
        unique_values = np.unique(data)
        print(f"🔢 Unique values in file: {unique_values}")
        assert set(unique_values) == {0, 1, 2, 3}, "Integer label values must be preserved exactly!"

    print("\n✨ VISIBILITY TEST PASSED! ✨")
    print("The masks now contain a high-contrast color palette while preserving the raw ID data.")
    
    # Cleanup
    test_path.unlink()

if __name__ == "__main__":
    test_mask_visibility()
