import json
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".nii.gz"}
SUPPORTED_LABEL_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".nii.gz"}
OVERLAY_COLORS = np.asarray(
    [
        [230, 57, 70],
        [29, 53, 87],
        [69, 123, 157],
        [42, 157, 143],
        [233, 196, 106],
        [244, 162, 97],
        [126, 87, 194],
        [76, 175, 80],
        [255, 112, 67],
        [38, 166, 154],
    ],
    dtype=np.uint8,
)


def case_id_from_path(path: Path, suffix_to_strip: str = "") -> str:
    stem = path.stem
    if stem.endswith(".nii"):
        stem = Path(stem).stem
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def collect_case_files(folder: Path, allowed_suffixes: set[str], suffix_to_strip: str = "") -> dict[str, Path]:
    mapping = {}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        # Special check for .nii.gz
        if str(path).lower().endswith(".nii.gz"):
            if ".nii.gz" not in allowed_suffixes:
                continue
        elif path.suffix.lower() not in allowed_suffixes:
            continue
            
        case_id = case_id_from_path(path, suffix_to_strip)
        if case_id in mapping:
            # If we have both .nii.gz and .png, we might get duplicates, let's prefer .nii.gz for now if it happens
            if path.suffix.lower() == ".gz":
                 mapping[case_id] = path
            continue
        mapping[case_id] = path
    if not mapping:
        raise FileNotFoundError(f"No compatible files were found under {folder}.")
    return mapping


def load_grayscale_image(path: Path) -> np.ndarray:
    if str(path).lower().endswith(".nii.gz"):
        return np.asarray(nib.load(str(path)).get_fdata()).astype(np.float32).squeeze()
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def load_label_image(path: Path) -> np.ndarray:
    if str(path).lower().endswith(".nii.gz"):
        return np.asarray(nib.load(str(path)).get_fdata()).astype(np.int32).squeeze()
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D label image but got shape {array.shape} for {path}")
    return array


def get_pascal_voc_palette() -> list[int]:
    """Generates a standard distinct color palette for label masks."""
    palette = [0, 0, 0]  # Background is black
    for j in range(1, 256):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
        palette.extend([r, g, b])
    return palette


def save_label_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).lower().endswith(".nii.gz"):
        nib.save(nib.Nifti1Image(array.astype(np.int32), np.eye(4)), str(path))
        return
    
    # Apply color palette to PNG/BMP to fix visibility of low-integer labels
    max_value = int(array.max()) if array.size else 0
    # Create PIL image in 'P' (Palette) mode
    img = Image.fromarray(array.astype(np.uint8, copy=False), mode="P")
    img.putpalette(get_pascal_voc_palette())
    img.save(path)


def normalize_preview_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32, copy=False)
    if image.size == 0:
        return image.astype(np.uint8)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def colorize_ordered_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros(mask.shape + (3,), dtype=np.uint8)
    positive_labels = sorted(int(label) for label in np.unique(mask) if label > 0)
    for color_index, label in enumerate(positive_labels):
        color[mask == label] = OVERLAY_COLORS[color_index % len(OVERLAY_COLORS)]
    return color


def save_overlay_preview(
    image_path: Path,
    ordered_mask: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
    annotations: list[dict] | None = None,
) -> None:
    """
    Saves a clinical preview overlay.
    
    Args:
        image_path: Path to the source X-ray image.
        ordered_mask: The label mask to overlay.
        output_path: Path where the preview will be saved.
        alpha: Transparency of the mask overlay.
        annotations: List of dicts with {"text": str, "centroid": (y, x)} for text labels.
    """
    base = normalize_preview_image(load_grayscale_image(image_path))
    rgb = np.repeat(base[..., None], 3, axis=2)
    color = colorize_ordered_mask(ordered_mask)
    positive = ordered_mask > 0
    overlay_array = rgb.copy()
    overlay_array[positive] = (
        (1.0 - alpha) * rgb[positive].astype(np.float32) + alpha * color[positive].astype(np.float32)
    ).astype(np.uint8)
    
    img = Image.fromarray(overlay_array)
    
    if annotations:
        draw = ImageDraw.Draw(img)
        # Try to load a professional font, fall back to default
        try:
            # Scale font size relative to image height (approx 3% of height)
            font_size = max(20, int(img.height * 0.03))
            font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        for ann in annotations:
            text = str(ann["text"])
            y, x = ann["centroid"]
            
            # Use anchor='mm' (middle-middle) for perfect centering if supported, 
            # otherwise calculate bounding box
            try:
                # High-contrast style: White text with professional black outline
                draw.text(
                    (x, y), 
                    text, 
                    fill=(255, 255, 255), 
                    font=font, 
                    anchor="mm",
                    stroke_width=2,
                    stroke_fill=(0, 0, 0)
                )
            except Exception:
                # Fallback for older PIL versions
                w, h = draw.textsize(text, font=font)
                draw.text((x - w/2, y - h/2), text, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
