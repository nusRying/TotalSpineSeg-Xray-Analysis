import os
import logging
from pathlib import Path
import numpy as np
import nibabel as nib

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def find_files(directory: Path, pattern: str) -> list[Path]:
    return list(directory.rglob(pattern))

def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, any]:
    img = nib.load(str(path))
    return img.get_fdata(), img.affine, img.header

def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path, dtype=None):
    if dtype:
        data = data.astype(dtype)
    img = nib.nifti1.Nifti1Image(data, affine)
    nib.save(img, str(path))

def image_to_nifti(data: np.ndarray, path: Path, spacing: tuple = (1.0, 1.0)):
    affine = np.diag([spacing[0], spacing[1], 1.0, 1.0])
    img = nib.nifti1.Nifti1Image(data, affine)
    nib.save(img, str(path))

def normalize_xray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        return (image - img_min) / (img_max - img_min)
    return image

def window_ct(volume: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    return np.clip(volume, img_min, img_max)
