# TotalSpineSeg X-Ray Project Memory

Last updated: `2026-04-06`

This document records the full working state of the X-ray adaptation effort in this repository so future work can resume without reconstructing context from chat history.

## 1. Project Goal

Original client ask:

- adapt an existing 3D framework for use with 2D X-ray images
- use Python and PyTorch
- focus on segmentation
- expected milestone framing:
  - milestone 1: design the adapted architecture using public X-ray data
  - milestone 2: develop and test the adapted architecture using public X-ray data

## 2. Key Scope Decisions

The original `totalspineseg` repository is a 3D MRI pipeline, not a generic 2D X-ray package.

Important repository characteristics before adaptation:

- 3D MRI / NIfTI oriented
- MRI-specific preprocessing and resampling
- two-stage nnU-Net flow
- MRI-target outputs such as vertebrae, discs, spinal cord, and spinal canal

Because of modality mismatch, the X-ray adaptation was narrowed intentionally.

Final baseline scope:

- modality: `2D AP spine X-ray`
- architecture: `nnU-Net 2d`
- task: binary vertebra segmentation
- output labels:
  - `0`: background
  - `1`: vertebrae
- anatomical scope for the trained baseline: `thoracic + lumbar (T1-L5)`

Explicitly not covered in the current baseline:

- cervical vertebrae
- spinal cord
- spinal canal
- MRI-style step1/step2 reproduction
- vertebra-wise anatomical naming guarantees

## 3. Anatomy Coverage Decision

This point matters because it affects what can honestly be claimed to the client.

Current trained data covers:

- `T1-T12`
- `L1-L5`

That means:

- thoracic: covered
- lumbar: covered
- cervical: not covered

Reason:

- the available public data used in this work does not include `C1-C7` annotations

Correct statement to client:

- current working version is a `thoracolumbar` baseline
- full `cervical + thoracic + lumbar` coverage requires an additional cervical-labeled dataset and retraining

## 4. Milestone 1 Outcome

Milestone 1 produced the design and scaffolding for a realistic X-ray baseline.

Main milestone 1 files:

- [xray_milestone1_design.md](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/docs/xray_milestone1_design.md)
- [xray_milestone1_design.pdf](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/docs/xray_milestone1_design.pdf)
- [xray_landmarks_to_mask.py](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/scripts/xray_landmarks_to_mask.py)
- [prepare_xray_dataset.py](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/scripts/prepare_xray_dataset.py)
- [train_xray.sh](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/scripts/train_xray.sh)

What milestone 1 established:

- a 2D X-ray path inside the repository
- a landmark-to-mask conversion step
- an nnU-Net v2 raw dataset builder
- a realistic scope note documenting why MRI parity is not a short-term target

## 5. Milestone 2 Outcome

Milestone 2 is complete for the scoped baseline.

Main milestone 2 files:

- [xray_milestone2_workflow.md](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/docs/xray_milestone2_workflow.md)
- [common.py](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/totalspineseg/xray/common.py)
- [postprocess.py](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/totalspineseg/xray/postprocess.py)
- [evaluate.py](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/totalspineseg/xray/evaluate.py)
- [inference.py](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/totalspineseg/xray/inference.py)
- [train_xray.ps1](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/scripts/train_xray.ps1)

What milestone 2 added:

- X-ray inference wrapper
- connected-component cleanup
- ordered-mask generation
- preview overlay generation
- binary evaluation reports
- Windows training wrapper for nnU-Net

## 6. Public Dataset Used

Local dataset root:

- [vertebrae_for_scoliosis](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray/vertebrae_for_scoliosis)

Extracted folders:

- [raw](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray/vertebrae_for_scoliosis/extracted/raw)
- [masks](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray/vertebrae_for_scoliosis/extracted/masks)

Observed counts:

- images: `737`
- masks: `737`

Important note:

- this was a practical public segmentation dataset used to produce a working baseline
- it is not the official client phrasing of `609 training + 159 testing images with 18 landmark points`
- therefore, the implementation is valid as a working baseline, but the exact dataset split differs from the original wording in the client text

## 7. nnU-Net Dataset Built

Raw nnU-Net dataset:

- [Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/raw/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP)

Preprocessed dataset:

- [preprocessed dataset root](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/preprocessed/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP)

Key dataset metadata:

- dataset id: `201`
- dataset name: `Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP`
- channel: `XRay`
- labels:
  - `background: 0`
  - `vertebrae: 1`
- file ending: `.png`
- `numTraining` in dataset.json: `590`

Dataset files:

- [dataset.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/raw/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/dataset.json)
- [xray_split.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/raw/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/xray_split.json)
- [dataset_fingerprint.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/preprocessed/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/dataset_fingerprint.json)
- [nnUNetPlans.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/preprocessed/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/nnUNetPlans.json)
- [nnUNetPlans_4GB.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/preprocessed/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/nnUNetPlans_4GB.json)

Actual train/test counts used:

- training: `590`
- test: `147`

## 8. Environment

Environment spec:

- [environment-xray.yml](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/environment-xray.yml)

Conda environment name:

- `totalspineseg-xray`

Conda executable used on this machine:

- `C:\Users\umair\miniconda3\Scripts\conda.exe`

Important environment notes:

- the env initially used CPU-only PyTorch
- the laptop has an `NVIDIA GeForce RTX 3050 Laptop GPU`
- PyTorch was replaced with a CUDA build so training could use the GPU

Working CUDA install used:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 9. Training Configuration

Training wrapper:

- [train_xray.ps1](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/scripts/train_xray.ps1)

Configuration actually used:

- dataset: `201`
- configuration: `2d`
- trainer: `nnUNetTrainer`
- planner: `ExperimentPlanner`
- plans: `nnUNetPlans_4GB`
- fold: `0`
- device: `cuda`

Reason for custom plan:

- laptop GPU has about `4 GB` VRAM
- the custom 4 GB plan avoids an oversized default plan

Results folder:

- [fold_0 results](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/results/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/nnUNetTrainer__nnUNetPlans_4GB__2d/fold_0)

Important files in the fold result:

- [checkpoint_best.pth](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/results/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/nnUNetTrainer__nnUNetPlans_4GB__2d/fold_0/checkpoint_best.pth)
- [checkpoint_latest.pth](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/results/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/nnUNetTrainer__nnUNetPlans_4GB__2d/fold_0/checkpoint_latest.pth)
- [progress.png](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/results/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/nnUNetTrainer__nnUNetPlans_4GB__2d/fold_0/progress.png)
- [debug.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/results/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/nnUNetTrainer__nnUNetPlans_4GB__2d/fold_0/debug.json)
- [training_log_2026_4_5_02_28_05.txt](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/nnUNet/results/Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP/nnUNetTrainer__nnUNetPlans_4GB__2d/fold_0/training_log_2026_4_5_02_28_05.txt)

Training stop condition:

- training was manually stopped after a strong baseline had been reached
- there is no `checkpoint_final.pth`
- inference must therefore use `checkpoint_best.pth`

Observed late training behavior:

- pseudo Dice stabilized around `0.85-0.86`
- by epoch `136`, validation pseudo Dice remained in the expected band

## 10. Final Tested Inference/Evaluation Outputs

Primary clean wrapper output:

- [fold0_4gb_wrapper](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper)

Contains:

- [binary](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper/binary)
- [ordered](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper/ordered)
- [preview](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper/preview)
- [report](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper/report)
- [inference_manifest.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper/inference_manifest.json)
- [postprocess_summary.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper/postprocess_summary.json)

Metrics:

- [metrics_summary.json](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper/report/metrics_summary.json)
- [per_case_metrics.csv](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/data/xray_inference/fold0_4gb_wrapper/report/per_case_metrics.csv)

Final measured test results:

- num cases: `147`
- mean Dice: `0.8560225623330332`
- mean IoU: `0.7524656978245925`
- mean precision: `0.8527379077195671`
- mean recall: `0.8677525455725534`
- missing predictions: `0`
- missing references: `0`

This is the baseline result that was shared as the milestone-2 completion result.

## 11. Packaging for Client Testing

Client-facing delivery folder:

- [totalspineseg_xray_test_package](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/deliverables/totalspineseg_xray_test_package)

Zip archive:

- [totalspineseg_xray_test_package.zip](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/deliverables/totalspineseg_xray_test_package.zip)

Package contents:

- checkpoint
- environment file
- package metadata
- X-ray inference code
- sample input images
- sample output overlays
- metrics summary
- short README
- PowerShell test runner

Important package files:

- [package README.md](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/deliverables/totalspineseg_xray_test_package/README.md)
- [RUN_TEST.ps1](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/deliverables/totalspineseg_xray_test_package/RUN_TEST.ps1)
- [checkpoint_best.pth](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/deliverables/totalspineseg_xray_test_package/checkpoint_best.pth)

Archive size:

- about `345 MB`

## 12. Important Code Fixes Made During This Work

### 12.1 Package discovery fix

File:

- [pyproject.toml](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/pyproject.toml)

Fix:

- package discovery was constrained with `include = ["totalspineseg*"]`
- this avoided setuptools confusion with top-level folders such as `data/`

### 12.2 Grayscale staging fix

File:

- [prepare_xray_dataset.py](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/scripts/prepare_xray_dataset.py)

Fix:

- staged X-rays are explicitly converted to grayscale with `image.convert("L")`
- this was required because nnU-Net rejected 3-channel PNGs for the single-channel 2D task

### 12.3 Windows training wrapper

File:

- [train_xray.ps1](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/scripts/train_xray.ps1)

Fixes/features:

- auto-sets repo-local nnU-Net directories
- supports GPU memory target
- supports 4 GB planning flow
- detects `data_identifier` from the plans JSON
- fail-fast wrapper around each nnU-Net command

### 12.4 X-ray inference wrapper fixes

File:

- [inference.py](/c:/Users/umair/Videos/Freelance/Sunshine%20V2/totalspineseg/xray/inference.py)

Fixes made:

- added checkpoint selection support with `--checkpoint`
- made the wrapper prefer the active Conda env’s `nnUNetv2_predict` executable
- stopped accidental double suffix staging such as `_0000_0000`
- made the wrapper inject the repo-local `data/nnUNet` paths into the nnU-Net subprocess automatically

These fixes were necessary because:

- training was stopped before `checkpoint_final.pth` existed
- Windows path resolution initially picked a global roaming-Python nnU-Net executable
- staged file names initially mismatched the predicted file names

## 13. Known Warnings and Non-Blocking Issues

### Requests warning

Repeated warning seen in shell output:

- `RequestsDependencyWarning: urllib3 ... doesn't match a supported version`

Source:

- roaming Python packages under `C:\Users\umair\AppData\Roaming\Python\Python313\...`

Impact:

- did not block training, inference, evaluation, or packaging

### Hiddenlayer warning

Observed message:

- `Unable to plot network architecture: No module named 'hiddenlayer'`

Impact:

- harmless
- only affects architecture plotting, not model training or inference

## 14. Commands That Matter

### Activate environment

```powershell
C:\Users\umair\miniconda3\Scripts\conda.exe activate totalspineseg-xray
```

### Training command used

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\umair\Videos\Freelance\Sunshine V2\scripts\train_xray.ps1" -Dataset 201 -Fold 0 -nnUNetPlans nnUNetPlans_4GB -GpuMemoryTargetGb 4
```

### Working inference command

```powershell
python -m totalspineseg.xray.inference `
  "C:\Users\umair\Videos\Freelance\Sunshine V2\data\nnUNet\raw\Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP\imagesTs" `
  "C:\Users\umair\Videos\Freelance\Sunshine V2\data\xray_inference\fold0_4gb_wrapper" `
  --dataset-id 201 `
  --configuration 2d `
  --trainer nnUNetTrainer `
  --plans nnUNetPlans_4GB `
  --checkpoint checkpoint_best.pth `
  --folds 0 `
  --device cuda `
  --min-area 64 `
  --overwrite
```

### Evaluation command

```powershell
python -m totalspineseg.xray.evaluate `
  "C:\Users\umair\Videos\Freelance\Sunshine V2\data\xray_inference\fold0_4gb_wrapper\binary" `
  "C:\Users\umair\Videos\Freelance\Sunshine V2\data\nnUNet\raw\Dataset201_TotalSpineSeg_XRay_Thoracolumbar_AP\labelsTs" `
  "C:\Users\umair\Videos\Freelance\Sunshine V2\data\xray_inference\fold0_4gb_wrapper\report"
```

## 15. Client Communication Memory

Correct client-facing statement at the current project state:

- there is a working X-ray baseline ready for testing
- it supports `thoracic + lumbar (T1-L5)`
- it does not yet support cervical
- the current held-out baseline result is about `0.856` mean Dice
- full `cervical + thoracic + lumbar` support requires additional cervical-labeled data and another training pass

What should not be claimed:

- full 3-level coverage is complete
- full clinical validation is complete
- MRI parity has been achieved

## 16. Recommended Next Steps

If continuing toward the original client ambition, the logical next step is:

1. obtain a cervical-labeled AP X-ray dataset with `C1-C7`
2. define a unified target for `cervical + thoracic + lumbar`
3. retrain the model with merged or staged data
4. decide whether the output should remain binary segmentation or move to vertebra-wise region labels
5. produce a second evaluation report for the expanded scope

## 17. Current Truth Summary

Current truth, in one sentence:

This repository now contains a working 2D AP thoracolumbar X-ray segmentation baseline inside the `totalspineseg` project, trained and evaluated on public data, packaged for client testing, with a held-out mean Dice of about `0.856`, but it does not yet include cervical coverage.
