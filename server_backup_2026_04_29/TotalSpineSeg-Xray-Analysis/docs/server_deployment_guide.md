# TotalSpineSeg X-Ray — Complete Server Deployment Guide
## "End Goal" Clinical Training Run | A100 Server Edition

**Project:** TotalSpineSeg-Xray-Analysis (`nusRying/TotalSpineSeg-Xray-Analysis`)
**Target Dataset:** Dataset 202 — TotalSpineSeg_XRay_FullSpine_AP_Lat
**Target Images:** ~15,500 (AASCE + VinDr-SpineXR + CSXA V3.0)
**Code Commit:** `f85fc58` — Clinical Geometry Engine + Finalized Strategy

---

## ⚠️ MANDATORY FIRST STEP

Ali, please run this exact command to ensure you have the latest production logic:
```bash
cd ~/TotalSpineSeg-Xray-Analysis
git fetch origin
git checkout main
git pull origin main
git log --oneline -1 # MUST output: f85fc58
```

---

## PHASE 1: Build the Environment

```bash
# 1.1 Install system-level medical libraries
pip install pydicom scikit-image scipy scikit-learn pillow numpy nibabel tqdm nnunetv2

# 1.2 Verify GPU is visible to PyTorch
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.get_device_name(0))"
# EXPECTED: CUDA: True | GPUs: NVIDIA A100...

# 1.3 Setup Persistence (so variables survive disconnects)
echo 'export TOTALSPINESEG_DATA="'$HOME'/TotalSpineSeg-Xray-Analysis/data"' >> ~/.bashrc
echo 'export nnUNet_raw="$TOTALSPINESEG_DATA/nnUNet/raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="$TOTALSPINESEG_DATA/nnUNet/preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="$TOTALSPINESEG_DATA/nnUNet/results"' >> ~/.bashrc
source ~/.bashrc
```

---

## PHASE 2: Automated Dataset Acquisition

We have unified all downloads into a single, resume-supported script.

```bash
cd ~/TotalSpineSeg-Xray-Analysis

# Run the downloader (This handles the 36GB VinDr + 3GB CSXA automatically)
# Use your PhysioNet credentials (umairejaz04 / Umair@825)
python3 scripts/fetch_xray_datasets.py \
  --output-dir "$TOTALSPINESEG_DATA" \
  --user umairejaz04 \
  --password "Umair@825"
```

---

## PHASE 3: Anatomical Mask Generation

> [!IMPORTANT]
> **C1 DISCLAIMER:** We are excluding the C1 Atlas because no public X-ray dataset exists for it. We will train on C2–S1. This matches the client's reference images which showed C1 as "uncertain" (`T1?`).

```bash
cd ~/TotalSpineSeg-Xray-Analysis

# 3.1 Convert CSXA to Canonical CSV
python3 scripts/map_csxa_landmarks.py \
  --json-dir "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/annotations/" \
  --output-csv "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/csxa_landmarks.csv"

# 3.2 Combine Images and Generate Anatomical (Bone-Shaped) Masks
# Note: Use --anatomical to avoid simple rectangles.
python3 scripts/xray_landmarks_to_mask.py \
    --images-dir "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/images" \
    --annotations "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/csxa_landmarks.csv" \
    --output-dir "$TOTALSPINESEG_DATA/xray/masks" \
    --mode label --ordered-labels "C2-C7" --anatomical --overwrite

# [Repeat for VinDr-SpineXR images/annotations similarly]
```

---

## PHASE 4: Final Training Launch (A100)

```bash
# 4.1 Create the nnUNet Dataset
python3 scripts/prepare_xray_dataset.py \
  --images-dir "$TOTALSPINESEG_DATA/xray/images" \
  --labels-dir "$TOTALSPINESEG_DATA/xray/masks" \
  --output-root "$nnUNet_raw" --dataset-id 202

## PHASE 5: Launch Training (Resume Mode)

To save time and achieve 96%+ accuracy faster, we recommend initializing the training with the Milestone 4 weights.

1.  **Manual Upload:** Upload the `weights/` folder from the Milestone 4 backup to the server.
2.  **Start Training:** Use the `-pretrained_weights` flag.

```bash
# Inside tmux
cd ~/TotalSpineSeg-Xray-Analysis

# 5.1 Plan and Preprocess
nnUNetv2_plan_and_preprocess -d 202 -c 2d

# 5.2 Start Training (Pretrained Mode)
# Replace /path/to/checkpoint_final.pth with the actual uploaded path
nnUNetv2_train 202 2d 0 -pretrained_weights /path/to/checkpoint_final.pth
```

---

## PHASE 5: Clinical Results Verify

Once training finishes, run this to see if Cobb's Angles are working:
```bash
totalspineseg_xray_inference data/test.png data/output --dataset-id 202
cat data/output/postprocess_summary.json | grep -A 5 "clinical_metrics"
```
