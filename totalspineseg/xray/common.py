import json
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
SUPPORTED_LABEL_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}
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
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def collect_case_files(folder: Path, allowed_suffixes: set[str], suffix_to_strip: str = "") -> dict[str, Path]:
    mapping = {}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        case_id = case_id_from_path(path, suffix_to_strip)
        if case_id in mapping:
            raise ValueError(f'Duplicate case id "{case_id}" found under {folder}.')
        mapping[case_id] = path
    if not mapping:
        raise FileNotFoundError(f"No compatible files were found under {folder}.")
    return mapping


def load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def load_label_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D label image but got shape {array.shape} for {path}")
    return array


def save_label_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_value = int(array.max()) if array.size else 0
    dtype = np.uint8 if max_value < 256 else np.uint16
    Image.fromarray(array.astype(dtype, copy=False)).save(path)


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


def save_overlay_preview(image_path: Path, ordered_mask: np.ndarray, output_path: Path, alpha: float = 0.45) -> None:
    base = normalize_preview_image(load_grayscale_image(image_path))
    rgb = np.repeat(base[..., None], 3, axis=2)
    color = colorize_ordered_mask(ordered_mask)
    positive = ordered_mask > 0
    overlay = rgb.copy()
    overlay[positive] = (
        (1.0 - alpha) * rgb[positive].astype(np.float32) + alpha * color[positive].astype(np.float32)
    ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
