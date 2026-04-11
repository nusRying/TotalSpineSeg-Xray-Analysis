"""
src/generate_phantom.py

Generates synthetic spine phantom NIfTI data (CT-like volumes + vertebra masks)
so the pipeline can be fully tested without downloading real datasets.

Usage:
    python src/generate_phantom.py --output data/raw/verse_sample --num_cases 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from utils import ensure_dir, save_nifti, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phantom geometry constants
# ---------------------------------------------------------------------------

VOLUME_SHAPE = (120, 120, 256)   # (X, Y, Z) — axial slices along Z
VOXEL_SPACING = (1.0, 1.0, 1.0)  # mm

# Vertebra parameters: (z_center, half_height, half_width, label_id)
# Mapping to TSS: C1=11, C2=12... T1=21... L1=41... Sacrum=50
PHANTOM_VERTEBRAE = [
    (240, 5, 12, 11),  # C1
    (225, 5, 12, 12),  # C2
    (210, 5, 12, 13),  # C3
    (195, 5, 12, 14),  # C4
    (180, 5, 13, 15),  # C5
    (165, 5, 13, 16),  # C6
    (150, 5, 13, 17),  # C7
    (135, 6, 14, 21),  # T1
    (120, 6, 14, 22),  # T2
    (105, 6, 15, 23),  # T3
    (90,  6, 15, 24),  # T4
    (75,  7, 16, 25),  # T5
    (60,  7, 16, 41),  # L1
    (45,  8, 17, 42),  # L2
    (30,  8, 18, 43),  # L3
    (15,  9, 20, 50),  # Sacrum
]

IVD_GAP = 8  # voxels gap between vertebrae (disc space)


def _make_ellipsoid_mask(
    shape: tuple,
    center: tuple,
    radii: tuple,
) -> np.ndarray:
    """Return a boolean 3D mask for an axis-aligned ellipsoid."""
    cx, cy, cz = center
    rx, ry, rz = radii
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    dist = (
        ((X - cx) / rx) ** 2
        + ((Y - cy) / ry) ** 2
        + ((Z - cz) / rz) ** 2
    )
    return dist <= 1.0


def generate_phantom_volume(
    shape: tuple = VOLUME_SHAPE,
    noise_level: float = 0.05,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic CT-like volume and corresponding vertebra label mask.

    Returns:
        ct_volume   : float32 array, shape (X, Y, Z), HU-like values
        label_mask  : int16 array,   shape (X, Y, Z), vertebra label IDs
    """
    if rng is None:
        rng = np.random.default_rng()

    X, Y, Z = shape
    cx, cy = X // 2, Y // 2

    ct = np.full(shape, -1000.0, dtype=np.float32)  # air background
    label = np.zeros(shape, dtype=np.int16)

    # Soft-tissue body cylinder
    for xi in range(X):
        for yi in range(Y):
            dist = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
            if dist < 28:
                ct[xi, yi, :] = rng.normal(50, 20, size=Z).astype(np.float32)

    # Spinal canal (low-density cylinder, posterior)
    canal_cx, canal_cy = cx, cy + 12
    for xi in range(X):
        for yi in range(Y):
            dist = np.sqrt((xi - canal_cx) ** 2 + (yi - canal_cy) ** 2)
            if dist < 4:
                ct[xi, yi, :] = rng.normal(10, 5, size=Z).astype(np.float32)

    # Vertebrae (high-density ellipsoids)
    for (z_center, half_h, half_w, label_id) in PHANTOM_VERTEBRAE:
        mask = _make_ellipsoid_mask(
            shape,
            center=(cx, cy - 5, z_center),
            radii=(half_w, half_w * 0.6, half_h),
        )
        ct[mask] = rng.normal(700, 80, size=mask.sum()).astype(np.float32)
        label[mask] = label_id

    # Add noise
    noise = rng.normal(0, noise_level * 1000, size=shape).astype(np.float32)
    ct += noise
    ct = np.clip(ct, -1024, 3071)

    return ct, label


def generate_and_save(
    output_dir: Path,
    num_cases: int = 3,
    seed: int = 42,
) -> None:
    """Generate `num_cases` phantom volumes and save as NIfTI."""
    ensure_dir(output_dir)
    rng = np.random.default_rng(seed)

    for i in range(num_cases):
        case_id = f"sub-phantom{i + 1:03d}"
        logger.info("Generating phantom: %s", case_id)

        # Slight variation per case
        noise = 0.04 + rng.uniform(-0.01, 0.02)
        ct_vol, label_vol = generate_phantom_volume(noise_level=noise, rng=rng)

        affine = np.diag([*VOXEL_SPACING, 1.0])

        ct_path = output_dir / f"{case_id}_dir-sag_ct.nii.gz"
        lbl_path = output_dir / f"{case_id}_dir-sag_seg-vert_msk.nii.gz"

        save_nifti(ct_vol, affine, ct_path, dtype=np.float32)
        save_nifti(label_vol, affine, lbl_path, dtype=np.int16)

        logger.info("  Saved: %s", ct_path.name)
        logger.info("  Saved: %s", lbl_path.name)
        logger.info(
            "  CT shape: %s | Vertebrae: %d",
            ct_vol.shape,
            len(np.unique(label_vol)) - 1,
        )

    logger.info("Phantom generation complete. %d cases in %s", num_cases, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic spine phantom data for pipeline testing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/verse_sample"),
        help="Output directory for phantom NIfTI files.",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=3,
        help="Number of phantom cases to generate (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    generate_and_save(args.output, num_cases=args.num_cases, seed=args.seed)


if __name__ == "__main__":
    main()
