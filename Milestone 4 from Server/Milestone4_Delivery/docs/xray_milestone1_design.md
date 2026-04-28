# TotalSpineSeg X-Ray Milestone 1

## Scope

Milestone 1 is intentionally narrower than the MRI pipeline in this repository.
The current TotalSpineSeg codebase is built around:

- 3D MRI volumes
- NIfTI preprocessing and resampling
- a two-stage nnU-Net pipeline
- postprocessing for vertebrae, discs, cord, and canal labeling

That design does not transfer directly to 2D X-ray images. The milestone 1 target is therefore:

- modality: 2D spine X-ray
- view: AP thoracolumbar radiographs
- public data: AASCE / SpineWeb style landmark annotations
- model task: vertebrae body segmentation baseline
- architecture: single-stage `nnU-Net 2d`

This is a proof-of-concept baseline for X-ray. It does not attempt to reproduce MRI-only targets such as spinal cord or spinal canal segmentation.

## Dataset Choice

Primary public-data target:

- AASCE 2019 challenge / SpineWeb Dataset 16

Reason for choosing it:

- public AP scoliosis X-rays with `T1-L5` landmark coverage
- established benchmark for scoliosis assessment
- landmark supervision can be converted into vertebral body masks

Anatomical coverage note:

- if the source dataset is the common `T1-L5` scoliosis landmark set, it covers the thoracic and lumbar spine only
- `T1-L5` does not include the cervical spine because cervical vertebrae are `C1-C7`
- a `T1-L5` dataset contains 17 vertebrae in total: `T1-T12` and `L1-L5`
- if cervical coverage is required, we need either a second cervical X-ray dataset or a different dataset that explicitly labels `C1-C7`

Fallback public-data option if access or label export differs:

- any AP thoracolumbar X-ray dataset that can be normalized into the same landmark CSV schema used by `scripts/xray_landmarks_to_mask.py`

## Target Labels

Milestone 1 uses a binary foreground target:

- `0`: background
- `1`: vertebrae

Why binary first:

- public X-ray labels are usually landmarks, not dense masks for every anatomical structure
- binary vertebrae segmentation is the lowest-risk baseline
- it creates a usable first step for later vertebra ordering and numbering

Optional extension after the baseline:

- sequential vertebra labels derived from the same landmark file
- region labels such as cervical / thoracic / lumbar
- anatomical vertebra labels using official TotalSpineSeg ids such as `C1`, `T1`, `L5`

Important limitation for the current thoracolumbar dataset:

- thoracic and lumbar region labels can be learned directly from `T1-L5`
- cervical region labels cannot be learned from this dataset alone

## Architecture

Milestone 1 X-ray architecture:

- dataset: `Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP`
- input: single-channel grayscale X-ray image
- output: vertebrae segmentation mask
- nnU-Net configuration: `2d`
- file format: `.png`

Why `2d`:

- nnU-Net v2 supports native 2D raster images
- X-rays are projection images, not 3D volumes
- this avoids pseudo-3D conversions that are unnecessary for milestone 1

## Data Flow

1. Export or normalize the public landmark annotations into a canonical CSV file.
2. Run `scripts/xray_landmarks_to_mask.py` to rasterize vertebra polygons into segmentation masks.
   The script now supports binary masks, sequential masks, or anatomical labels via `--ordered-labels`, for example `T1-L5` or `C1-L5`.
3. Run `scripts/prepare_xray_dataset.py` to:
   - convert images and masks into lossless `.png`
   - create deterministic train/test splits
   - write an nnU-Net raw dataset
   - generate `dataset.json` with named labels when mask values match TotalSpineSeg vertebra ids
4. Run `scripts/train_xray.sh` to train a `2d` nnU-Net baseline.

## Canonical Landmark CSV Schema

Each row describes one vertebra polygon:

```text
case_id,vertebra_order,x1,y1,x2,y2,x3,y3,x4,y4
case_001,1,120,84,188,86,186,136,118,134
case_001,2,122,140,191,143,190,193,121,191
```

Notes:

- `case_id` must match the image stem
- `vertebra_order` is used to sort vertebrae superior-to-inferior
- coordinates may be absolute pixels or normalized floats, depending on the CLI flag

## Deliverables Covered By Milestone 1

- architecture decision for X-ray adaptation
- public-data choice and label strategy
- landmark-to-mask conversion script
- nnU-Net 2D dataset preparation script
- X-ray-specific training shell script

## Items Explicitly Deferred

Not part of milestone 1:

- spinal cord segmentation on X-ray
- spinal canal segmentation on X-ray
- MRI-style step1/step2 replication
- anatomical vertebra naming guarantees such as C7/T1/L5/S1
- full clinical validation

## Milestone 2 Start Point

Milestone 2 should build on this baseline by:

- training the first model on public AP thoracolumbar X-ray data
- running held-out evaluation
- adding simple connected-component cleanup
- adding vertebra ordering and optional numbering heuristics
