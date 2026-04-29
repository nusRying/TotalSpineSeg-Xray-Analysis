# TotalSpineSeg X-Ray: Milestone 3 Full-Spine Pipeline

This guide explains how to duplicate and use the TotalSpineSeg X-ray pipeline for full-spine (C1-S1) segmentation and labeling.

## 1. Environment Setup

### Prerequisites
- Python 3.10
- NVIDIA GPU (RTX 3050 4GB or better recommended)
- Conda or Mamba

### Installation
```powershell
# Create environment
conda create -n totalspineseg-xray python=3.10 -y
conda activate totalspineseg-xray

# Install Core Dependencies
pip install -r requirements-xray.txt

# Install PyTorch with CUDA 12.4 support (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install local package in editable mode
pip install -e .[nnunetv2]
```

## 2. Data Preparation

The Milestone 3 pipeline uses a **Hybrid Dataset** (Real + Synthetic) to achieve full clinical coverage.

### Real Data (C2-L5)
1. **Thoracolumbar (AASCE):** Place real images/masks in `data/xray/vertebrae_for_scoliosis/extracted`.
2. **Cervical (CSXA):** 
   - Download `datasets-JSON.zip` and `datasets-PNG.zip`.
   - Place in `data/xray/csxa/original`.
   - Map landmarks: `python scripts/map_csxa_landmarks.py --json-dir ... --output-csv ...`

### Synthetic Data (C1 & Sacrum)
To fill anatomical gaps, use the phantom generator:
```powershell
# Generate full-spine phantoms
python things_from_client/generate_phantom.py --num_cases 10 --output data/raw/phantom_full

# Generate AP and Lateral DRRs
python things_from_client/generate_drr.py --input data/raw/phantom_full --output data/processed/phantom_full_ap --projection ap
python things_from_client/generate_drr.py --input data/raw/phantom_full --output data/processed/phantom_full_lateral --projection lateral
```

### Final Consolidation
Run the preparation script to create the unified nnU-Net dataset:
```powershell
python scripts/prepare_xray_dataset.py --dataset-id 202 --dataset-name "TotalSpineSeg_XRay_FullSpine"
```

## 3. Training

The pipeline is optimized for 4GB VRAM GPUs (like the RTX 3050).

```powershell
# Launch training with 4GB VRAM optimization and Auto-Resume support
.\scripts\train_xray.ps1 -Dataset 202 -Fold 0 -GpuMemoryTargetGb 4
```

## 4. Inference & Labeling

Run inference on any X-ray image (AP or Lateral):
```powershell
totalspineseg_xray_inference input_image.png output_folder --ordered-labels "C1-L5"
```

## 5. Evaluation

Generate a clinically relevant report with per-region (Cervical, Thoracic, Lumbar, Sacrum) statistics:
```powershell
totalspineseg_xray_evaluate preds_dir ground_truth_dir report_dir --evaluation-mode multiclass
```

## 6. Using the Pre-trained Model (Included)

The delivery package includes the full trained nnU-Net export folder in `nnUNet_results/`. To use it for inference without retraining:

1. **Set Environment Variable:**
   ```powershell
   $env:nnUNet_results = "C:/path/to/extracted/nnUNet_results"
   ```
2. **Run Prediction:**
   Standard nnU-Net inference can now find the model automatically using Dataset ID 202.

---
**Maintained by:** TotalSpineSeg X-Ray Team
**Milestone 3 Status:** Full-Spine Ready
