"""
src/generate_drr.py

Digitally Reconstructed Radiograph (DRR) generation.

Converts 3D CT volumes (.nii.gz) into synthetic 2D X-ray images by summing
attenuation along the projection axis (simplified Beer-Lambert model).
Simultaneously projects 3D vertebra label masks down to 2D by taking the
label with maximum voxel coverage along the projection axis.

Usage:
    python src/generate_drr.py \\
        --input  data/raw/verse_sample \\
        --output data/processed/xray_synthetic \\
        --projection lateral   # or 'ap'

Output per case:
    <case_id>_xray_0000.nii.gz   — grayscale 2D X-ray image (nnU-Net channel 0)
    <case_id>_seg.nii.gz         — 2D vertebra label mask
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from utils import (
    ensure_dir,
    find_files,
    image_to_nifti,
    load_nifti,
    normalize_xray,
    setup_logging,
    window_ct,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

# Mapping of projection name to axis in (X, Y, Z) volume
PROJECTION_AXIS = {
    "lateral": 1,  # project along Y → lateral (sagittal) view
    "ap": 0,       # project along X → AP (coronal) view
}


def _hu_to_attenuation(volume_hu: np.ndarray) -> np.ndarray:
    """
    Convert HU values to linear attenuation coefficients (simplified).
    Air = -1000 HU → mu ≈ 0
    Water = 0 HU   → mu ≈ 0.019 /mm
    Bone = 700 HU  → mu ≈ 0.060 /mm
    Reference: Schneider et al. (1996)
    """
    mu = np.zeros_like(volume_hu, dtype=np.float32)
    # Soft tissue / fluid
    soft = (volume_hu > -300) & (volume_hu <= 100)
    mu[soft] = 0.019 + volume_hu[soft] * (0.024 - 0.019) / 400.0
    # Bone
    bone = volume_hu > 100
    mu[bone] = 0.024 + (volume_hu[bone] - 100) * (0.080 - 0.024) / 900.0
    mu[mu < 0] = 0
    return mu


def generate_drr(
    ct_volume: np.ndarray,
    axis: int,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Generate a DRR by summing attenuation along `axis`.

    Args:
        ct_volume: float32 array (X, Y, Z) in Hounsfield Units
        axis:      projection axis (0=X, 1=Y, 2=Z)
        spacing:   voxel spacing in mm

    Returns:
        drr: float32 2D array (H, W) normalized to [0, 1]
    """
    # Apply bone-window HU clipping for better vertebra contrast
    windowed = window_ct(ct_volume, window_center=400, window_width=1800)

    # Convert to attenuation
    mu = _hu_to_attenuation(windowed)

    # Integrate along projection axis
    dl = spacing[axis]
    projection = np.sum(mu, axis=axis) * dl  # Beer-Lambert integral

    # Invert: X-rays are bright where attenuation is LOW (air is bright)
    drr = np.exp(-projection)

    # Normalise to [0, 1]
    drr = normalize_xray(drr)

    # X-ray convention: bones appear dark → invert
    drr = 1.0 - drr

    return drr.astype(np.float32)


def project_label_mask(
    label_volume: np.ndarray,
    axis: int,
) -> np.ndarray:
    """
    Project a 3D label mask to 2D by majority vote along `axis`.

    For each 2D pixel, the label that occupies the most voxels along the
    projection ray is assigned. Background (0) is only assigned if no
    foreground voxel is present.

    Args:
        label_volume: int array (X, Y, Z)
        axis:         projection axis

    Returns:
        label_2d: int16 2D array
    """
    unique_labels = np.unique(label_volume)
    unique_labels = unique_labels[unique_labels > 0]  # skip background

    # Result shape after removing the projection axis
    result_shape = tuple(s for i, s in enumerate(label_volume.shape) if i != axis)
    label_2d = np.zeros(result_shape, dtype=np.int16)
    max_count = np.zeros(result_shape, dtype=np.int32)

    for lbl in unique_labels:
        binary = (label_volume == lbl).astype(np.int32)
        count = np.sum(binary, axis=axis)
        update = count > max_count
        label_2d[update] = lbl
        max_count[update] = count[update]

    return label_2d


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def process_case(
    ct_path: Path,
    label_path: Optional[Path],
    output_dir: Path,
    axis: int,
    case_id: str,
) -> bool:
    """
    Process one CT case → DRR + projected mask.

    Returns True on success, False on error.
    """
    try:
        logger.info("Processing case: %s", case_id)
        ct_vol, affine, header = load_nifti(ct_path)

        # Extract voxel spacing from NIfTI header
        try:
            zooms = header.get_zooms()
            spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        except Exception:
            spacing = (1.0, 1.0, 1.0)

        logger.debug("  CT shape: %s | spacing: %s", ct_vol.shape, spacing)

        # --- Generate DRR ---
        drr = generate_drr(ct_vol, axis=axis, spacing=spacing)
        logger.debug("  DRR shape: %s  range: [%.3f, %.3f]", drr.shape, drr.min(), drr.max())

        # Save DRR as 2D NIfTI (nnU-Net channel 0 → suffix _0000)
        drr_path = output_dir / f"{case_id}_0000.nii.gz"
        # Pixel spacing after projection (the two axes that were NOT summed)
        remaining_spacing = [s for i, s in enumerate(spacing) if i != axis]
        image_to_nifti(drr, drr_path, spacing=tuple(remaining_spacing))

        # --- Project label mask ---
        if label_path is not None and label_path.exists():
            label_vol, _, _ = load_nifti(label_path)
            label_vol = label_vol.astype(np.int16)
            label_2d = project_label_mask(label_vol, axis=axis)

            # Quick sanity: at least one vertebra should be visible
            n_labels = len(np.unique(label_2d)) - 1
            logger.debug("  Projected labels: %d vertebra classes visible", n_labels)

            mask_path = output_dir / f"{case_id}.nii.gz"
            image_to_nifti(label_2d.astype(np.float32), mask_path, spacing=tuple(remaining_spacing))
        else:
            logger.warning("  No label mask found for %s — skipping mask", case_id)

        return True

    except Exception as exc:
        logger.error("Failed to process case %s: %s", case_id, exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------

def process_dataset(
    input_dir: Path,
    output_dir: Path,
    projection: str = "lateral",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> None:
    """
    Process all CT cases in `input_dir` and split into train/val/test.

    Expected input naming convention (VerSe-compatible):
        <case_id>_dir-sag_ct.nii.gz       — CT volume
        <case_id>_dir-sag_seg-vert_msk.nii.gz — label mask (optional)
    """
    axis = PROJECTION_AXIS.get(projection.lower())
    if axis is None:
        raise ValueError(f"Unknown projection '{projection}'. Choose from: {list(PROJECTION_AXIS)}")

    # Find all CT files
    ct_files = find_files(input_dir, "*_ct.nii.gz")
    if not ct_files:
        # Fallback: any .nii.gz that doesn't look like a mask
        ct_files = [f for f in find_files(input_dir, "*.nii.gz") if "seg" not in f.name]

    if not ct_files:
        raise FileNotFoundError(f"No CT NIfTI files found in: {input_dir}")

    logger.info("Found %d CT cases in %s", len(ct_files), input_dir)

    # Split into train / val / test
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(ct_files))
    n_train = max(1, int(len(ct_files) * train_ratio))
    n_val = max(1, int(len(ct_files) * val_ratio))

    splits = {
        "train": [ct_files[i] for i in indices[:n_train]],
        "val":   [ct_files[i] for i in indices[n_train : n_train + n_val]],
        "test":  [ct_files[i] for i in indices[n_train + n_val :]],
    }

    # Edge case: ensure test is not empty
    if not splits["test"] and len(ct_files) >= 3:
        splits["test"] = [splits["val"].pop()]

    successes = 0
    failures = 0

    for split_name, split_files in splits.items():
        if not split_files:
            continue
        split_dir = ensure_dir(output_dir / split_name)

        for ct_path in split_files:
            # Derive case_id and matching label path — try multiple naming conventions
            case_id = ct_path.name.replace("_dir-sag_ct.nii.gz", "").replace("_ct.nii.gz", "")

            # Try all known label naming conventions in order
            label_candidates = [
                ct_path.parent / ct_path.name.replace("_dir-sag_ct.nii.gz", "_dir-sag_seg-vert_msk.nii.gz"),
                ct_path.parent / ct_path.name.replace("_ct.nii.gz", "_dir-sag_seg-vert_msk.nii.gz"),
                ct_path.parent / ct_path.name.replace("_ct.nii.gz", "_seg-vert_msk.nii.gz"),
                ct_path.parent / ct_path.name.replace("_ct.nii.gz", "_seg.nii.gz"),
                ct_path.parent / f"{case_id}_dir-sag_seg-vert_msk.nii.gz",
                ct_path.parent / f"{case_id}_seg.nii.gz",
            ]
            label_path = next((p for p in label_candidates if p.exists()), label_candidates[0])

            ok = process_case(
                ct_path=ct_path,
                label_path=label_path if label_path.exists() else None,
                output_dir=split_dir,
                axis=axis,
                case_id=case_id,
            )
            if ok:
                successes += 1
            else:
                failures += 1

    logger.info(
        "DRR generation complete. Success: %d | Failed: %d",
        successes,
        failures,
    )
    if failures == len(ct_files):
        raise RuntimeError("All cases failed during DRR generation. Check logs.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 2D X-rays (DRRs) from 3D CT volumes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing CT NIfTI files (.nii.gz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for DRR images and projected masks.",
    )
    parser.add_argument(
        "--projection",
        choices=["lateral", "ap"],
        default="lateral",
        help="Projection view: 'lateral' (default) or 'ap' (anteroposterior).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of cases for training (default: 0.70).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of cases for validation (default: 0.15).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        projection=args.projection,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
