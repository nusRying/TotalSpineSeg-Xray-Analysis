# TotalSpineSeg X-Ray — Complete Server Deployment Guide
## "End Goal" Clinical Training Run | A100 Server Edition

**Project:** TotalSpineSeg-Xray-Analysis (`nusRying/TotalSpineSeg-Xray-Analysis`)
**Target Dataset:** Dataset 202 — TotalSpineSeg_XRay_FullSpine_AP_Lat
**Target Images:** ~15,500 (AASCE + VinDr-SpineXR + CSXA V3.0)
**Expected Training Time:** 24–48 hours on A100 GPU
**Code Commit:** `2723a92` — Clinical Geometry Engine + DICOM Support

---

## TABLE OF CONTENTS

1. [Phase 0 — Pre-Flight System Check](#phase-0--pre-flight-system-check)
2. [Phase 1 — Environment Setup](#phase-1--environment-setup)
3. [Phase 2 — Dataset Acquisition](#phase-2--dataset-acquisition)
4. [Phase 3 — Ground Truth Mask Generation](#phase-3--ground-truth-mask-generation)
5. [Phase 4 — nnUNet Dataset Preparation & Training](#phase-4--nnunet-dataset-preparation--training)
6. [Phase 5 — Post-Training Clinical Verification](#phase-5--post-training-clinical-verification)
7. [Troubleshooting Reference](#troubleshooting-reference)

---

## PHASE 0 — Pre-Flight System Check

Run these checks **BEFORE** touching the project. Do not skip.

### 0.1 Verify GPU

```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...   Driver Version: ...   CUDA Version: 12.x                   |
|-------------------------------+----------------------+----------------------+
| GPU  0  A100-SXM4-40GB        | 00000000:00:1E.0 Off |                  0% |
+-----------------------------------------------------------------------------+
```

- If `nvidia-smi` is not found → GPU drivers are not installed. Stop.
- If CUDA Version is below 11.7 → nnUNet will not work. Update drivers.

### 0.2 Verify Python & PyTorch CUDA

```bash
python3 --version    # Must be 3.10 or higher

python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"
```

**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True
GPU name: NVIDIA A100-SXM4-40GB
```

> **If CUDA is False:** PyTorch was installed without GPU support.
> Fix: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### 0.3 Verify nnUNet Installation

```bash
which nnUNetv2_train
nnUNetv2_train --help 2>&1 | head -5
```

**Expected output:** Shows nnUNet usage help. If command not found:
```bash
pip install nnunetv2
```

### 0.4 Verify Available Disk Space

```bash
df -h ~
```

You need at least **80 GB free** to accommodate:

| Component | Size |
| :--- | :--- |
| VinDr-SpineXR raw data | ~36 GB |
| CSXA V3.0 | ~3 GB |
| Generated masks | ~5 GB |
| nnUNet preprocessed | ~15 GB |
| nnUNet results / checkpoints | ~10 GB |
| **Total** | **~69 GB** |

---

## PHASE 1 — Environment Setup

### 1.1 Clone or Update the Repository

```bash
# If the repository doesn't exist yet on the server:
git clone https://github.com/nusRying/TotalSpineSeg-Xray-Analysis.git
cd TotalSpineSeg-Xray-Analysis

# If it already exists, update it:
cd ~/TotalSpineSeg-Xray-Analysis
git fetch origin
git checkout main
git pull origin main
```

**Verify you have the correct commit:**
```bash
git log --oneline -3
```

**Expected first line:**
```
2723a92 feat: Add Clinical Geometry Engine, DICOM support, and Surgical JSON export
```

> If the commit is not there, the pull failed. Run `git pull --force origin main`.

### 1.2 Install All Python Dependencies

```bash
pip install \
    pydicom \
    scikit-image \
    scipy \
    scikit-learn \
    pillow \
    numpy \
    nibabel \
    tqdm \
    nnunetv2
```

**Verify the critical new modules load:**
```bash
python3 -c "from totalspineseg.xray.geometry import calculate_cobb_angle; print('Geometry Engine: OK')"
python3 -c "from totalspineseg.xray.postprocess import postprocess_prediction; print('Postprocess: OK')"
python3 -c "import pydicom; print('DICOM support: OK')"
```

**Expected output:**
```
Geometry Engine: OK
Postprocess: OK
DICOM support: OK
```

> If any line fails with `ImportError`, that package did not install. Re-run `pip install` for that specific package.

### 1.3 Configure Environment Variables (Persistent)

```bash
# Write to ~/.bashrc so they survive SSH disconnects and server reboots
cat >> ~/.bashrc << 'EOF'

# === TotalSpineSeg X-Ray Training ===
export TOTALSPINESEG_DATA="$HOME/TotalSpineSeg-Xray-Analysis/data"
export nnUNet_raw="$TOTALSPINESEG_DATA/nnUNet/raw"
export nnUNet_preprocessed="$TOTALSPINESEG_DATA/nnUNet/preprocessed"
export nnUNet_results="$TOTALSPINESEG_DATA/nnUNet/results"
export TOTALSPINESEG_DEVICE="cuda"
EOF

# Load variables into the current session immediately
source ~/.bashrc

# Verify all 4 are set — NONE should be blank
echo "DATA:         $TOTALSPINESEG_DATA"
echo "RAW:          $nnUNet_raw"
echo "PREPROCESSED: $nnUNet_preprocessed"
echo "RESULTS:      $nnUNet_results"
```

### 1.4 Create the Directory Structure

```bash
mkdir -p "$TOTALSPINESEG_DATA/raw_external/VinDr_SpineXR"
mkdir -p "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/images"
mkdir -p "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/annotations"
mkdir -p "$TOTALSPINESEG_DATA/xray/images"
mkdir -p "$TOTALSPINESEG_DATA/xray/masks"
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

echo "Directory structure created."
ls -la "$TOTALSPINESEG_DATA/"
```

---

## PHASE 2 — Dataset Acquisition

> **Target:** ~15,500 total images across 3 sources.

### 2.1 Check Existing AASCE Data (Baseline)

```bash
AASCE_IMG_COUNT=$(ls "$TOTALSPINESEG_DATA/xray/images/" 2>/dev/null | wc -l)
echo "AASCE images already present: $AASCE_IMG_COUNT"
```

- If count is 0: baseline data hasn't been set up. Check `scripts/prepare_datasets.sh`.
- If count > 0: this data will be upgraded with anatomical masks in Phase 3.

### 2.2 Download VinDr-SpineXR (~36 GB)

This is the backbone dataset covering the full spine (C1–S1) including pathological cases (fractures, osteophytes, scoliosis).

```bash
cd "$TOTALSPINESEG_DATA/raw_external/VinDr_SpineXR"

# Credentials: umairejaz04 / Umair@825
wget -r -N -c -np \
  --user umairejaz04 \
  --password "Umair@825" \
  https://physionet.org/files/vindr-spinexr/1.0.0/

echo "Download exit code: $?"
```

**Monitor download progress in a second terminal:**
```bash
watch -n 30 "du -sh $TOTALSPINESEG_DATA/raw_external/VinDr_SpineXR/"
```

**After download, verify:**
```bash
du -sh "$TOTALSPINESEG_DATA/raw_external/VinDr_SpineXR/"
# Expected: approx 33–36 GB

ls "$TOTALSPINESEG_DATA/raw_external/VinDr_SpineXR/physionet.org/files/vindr-spinexr/1.0.0/"
# Expected: folders named train/, test/, annotations/ or similar
```

**Common errors and fixes:**

| Error | Fix |
| :--- | :--- |
| `403 Forbidden` | Credentials wrong. Double-check username `umairejaz04` and password. |
| `Connection reset` | Unstable network. Re-run the same `wget` command — `-N -c` flags resume where it left off. |
| Very slow download | Normal; PhysioNet throttles bandwidth. Let it run overnight. |

### 2.3 Download CSXA V3.0 — Cervical Spine Atlas (~2.7 GB)

This dataset covers the cervical spine (C2–C7) with JSON landmark annotations.

```bash
cd "$TOTALSPINESEG_DATA/raw_external/CSXA_V3"

# Download PNG images (~1.3 GB)
wget -O datasets-PNG.zip \
  "https://china.scidb.cn/download?fileId=801011b2c734ad280b9326a29358730f"

# Download JSON annotations (~1.4 GB)
wget -O datasets-JSON.zip \
  "https://china.scidb.cn/download?fileId=5dada884dd8d622531e826f2452e35d7"

# Verify file sizes
ls -lh *.zip

# Extract
unzip -q datasets-PNG.zip -d images/
unzip -q datasets-JSON.zip -d annotations/

# Verify extraction
echo "CSXA images: $(find images/ -name '*.png' | wc -l)"
echo "CSXA JSONs:  $(find annotations/ -name '*.json' | wc -l)"
```

**Expected:** ~4,963 images and ~4,963 JSON files.

**If unzip fails (alternative method):**
```bash
python3 -c "import zipfile; zipfile.ZipFile('datasets-PNG.zip').extractall('images/')"
python3 -c "import zipfile; zipfile.ZipFile('datasets-JSON.zip').extractall('annotations/')"
```

---

## PHASE 3 — Ground Truth Mask Generation

> This phase uses our **Projective Template Warping** engine to convert raw
> landmark coordinates into high-fidelity anatomical bone masks.
> The `--anatomical` flag is **CRITICAL** — without it you get simple rectangles.

### 3.1 Convert CSXA JSON Landmarks → Canonical CSV

The CSXA dataset uses JSON. Our pipeline uses CSV. This step converts it:

```bash
cd ~/TotalSpineSeg-Xray-Analysis

python3 scripts/map_csxa_landmarks.py \
  --json-dir "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/annotations/" \
  --output-csv "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/csxa_landmarks.csv"

# Verify output
echo "CSV rows generated: $(wc -l < $TOTALSPINESEG_DATA/raw_external/CSXA_V3/csxa_landmarks.csv)"
head -3 "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/csxa_landmarks.csv"
```

**Expected CSV header:**
```
case_id,vertebra_order,x1,y1,x2,y2,x3,y3,x4,y4
```

**Expected row count:** ~25,000+ rows (up to 6 vertebrae × ~4,963 cases).

### 3.2 Copy CSXA Images Into the Unified Image Pool

```bash
find "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/images/" -name "*.png" \
  -exec cp {} "$TOTALSPINESEG_DATA/xray/images/" \;

echo "Total images after CSXA copy: $(ls $TOTALSPINESEG_DATA/xray/images/ | wc -l)"
```

### 3.3 Generate Anatomical Masks for CSXA (Cervical C2–C7)

```bash
python3 scripts/xray_landmarks_to_mask.py \
  --images-dir "$TOTALSPINESEG_DATA/xray/images" \
  --annotations "$TOTALSPINESEG_DATA/raw_external/CSXA_V3/csxa_landmarks.csv" \
  --output-dir "$TOTALSPINESEG_DATA/xray/masks" \
  --mode label \
  --ordered-labels "C2,C3,C4,C5,C6,C7" \
  --anatomical \
  --overwrite

echo "Masks after CSXA: $(ls $TOTALSPINESEG_DATA/xray/masks/ | wc -l)"
```

**Expected output:**
```
Wrote N masks to data/xray/masks using mode=label from N annotated cases.
```

### 3.4 Generate Anatomical Masks for VinDr-SpineXR

**Step 1 — Identify the annotation file:**
```bash
find "$TOTALSPINESEG_DATA/raw_external/VinDr_SpineXR" -name "*.csv" 2>/dev/null
VINDR_CSV=$(find "$TOTALSPINESEG_DATA/raw_external/VinDr_SpineXR" -name "*.csv" | head -1)
echo "Found: $VINDR_CSV"
head -2 "$VINDR_CSV"   # Check column names
```

**Step 2 — Copy VinDr images to unified pool:**
```bash
find "$TOTALSPINESEG_DATA/raw_external/VinDr_SpineXR" -name "*.png" \
  -exec cp {} "$TOTALSPINESEG_DATA/xray/images/" \;

echo "Total images after VinDr copy: $(ls $TOTALSPINESEG_DATA/xray/images/ | wc -l)"
```

**Step 3 — Generate masks:**
```bash
python3 scripts/xray_landmarks_to_mask.py \
  --images-dir "$TOTALSPINESEG_DATA/xray/images" \
  --annotations "$VINDR_CSV" \
  --output-dir "$TOTALSPINESEG_DATA/xray/masks" \
  --mode label \
  --ordered-labels "T1-L5" \
  --anatomical \
  --overwrite

echo "Masks after VinDr: $(ls $TOTALSPINESEG_DATA/xray/masks/ | wc -l)"
```

> **NOTE:** If you see `KeyError: 'case_id'`, the VinDr CSV uses different column names.
> Inspect with `head -1 $VINDR_CSV` then add:
> `--case-column <name> --point-columns xl yt xr yt xr yb xl yb`

### 3.5 Upgrade AASCE Masks with Anatomical Templates

This replaces the old rectangle-based masks with bone-shaped contours:

```bash
AASCE_CSV=$(find . -name "aasce_landmarks.csv" 2>/dev/null | head -1)
echo "AASCE CSV: $AASCE_CSV"

python3 scripts/xray_landmarks_to_mask.py \
  --images-dir "$TOTALSPINESEG_DATA/xray/images" \
  --annotations "$AASCE_CSV" \
  --output-dir "$TOTALSPINESEG_DATA/xray/masks" \
  --mode label \
  --ordered-labels "T1-L5" \
  --normalized \
  --anatomical \
  --overwrite
```

### 3.6 CRITICAL GATE — Final Dataset Count Verification

**Do NOT proceed to Phase 4 if this check fails.**

```bash
IMG_COUNT=$(ls "$TOTALSPINESEG_DATA/xray/images/" | wc -l)
MSK_COUNT=$(ls "$TOTALSPINESEG_DATA/xray/masks/" | wc -l)

echo "========================================"
echo "  FINAL DATASET VERIFICATION"
echo "========================================"
echo "  Total images: $IMG_COUNT"
echo "  Total masks:  $MSK_COUNT"
echo "========================================"

if [ "$IMG_COUNT" -eq "$MSK_COUNT" ] && [ "$IMG_COUNT" -gt 5000 ]; then
    echo "✅ PASS — Ready for Phase 4"
else
    echo "❌ FAIL — Fix before proceeding"
fi
```

**If counts don't match, find the orphaned images:**
```bash
comm -23 \
  <(ls "$TOTALSPINESEG_DATA/xray/images/" | sed 's/\.png//' | sort) \
  <(ls "$TOTALSPINESEG_DATA/xray/masks/" | sed 's/\.png//' | sort) \
  | head -20
```

---

## PHASE 4 — nnUNet Dataset Preparation & Training

### 4.1 Prepare the nnUNet v2 Dataset Structure

```bash
cd ~/TotalSpineSeg-Xray-Analysis

python3 scripts/prepare_xray_dataset.py \
  --images-dir "$TOTALSPINESEG_DATA/xray/images" \
  --labels-dir "$TOTALSPINESEG_DATA/xray/masks" \
  --output-root "$nnUNet_raw" \
  --dataset-id 202 \
  --dataset-name TotalSpineSeg_XRay_FullSpine_AP_Lat \
  --test-size 0.1 \
  --seed 42 \
  --overwrite

echo "Exit code: $?"
```

**Verify the dataset structure:**
```bash
DATASET_DIR="$nnUNet_raw/Dataset202_TotalSpineSeg_XRay_FullSpine_AP_Lat"

echo "=== Training / Test Case Counts ==="
echo "imagesTr: $(ls $DATASET_DIR/imagesTr/ | wc -l)"
echo "labelsTr: $(ls $DATASET_DIR/labelsTr/ | wc -l)"
echo "imagesTs: $(ls $DATASET_DIR/imagesTs/ | wc -l)"

echo ""
echo "=== dataset.json ==="
cat "$DATASET_DIR/dataset.json"
```

**Critical checks on `dataset.json`:**
- `numTraining` should be ≥ 9,000
- `file_ending` should be `.png`
- `labels` should contain `"background": 0` plus vertebra entries

### 4.2 Verify Dataset Integrity (nnUNet Built-in)

```bash
nnUNetv2_extract_fingerprint -d 202 -np 8 --verify_dataset_integrity
```

Expected: prints image size analysis. Any `ERROR` line means a corrupted mask. Find and delete/re-generate it.

### 4.3 Start Training Inside tmux — ESSENTIAL

> **tmux is mandatory.** Without it, if your SSH session drops, the training
> process dies and hours of A100 time are lost with no checkpoint.

```bash
# Install tmux if not present
apt-get install -y tmux 2>/dev/null

# Start a named session
tmux new-session -s spine_train

# Inside the tmux session — reload env vars first
source ~/.bashrc

# Run the full training pipeline (handles fingerprint → planning → preprocess → train)
bash ~/TotalSpineSeg-Xray-Analysis/scripts/train_xray.sh 202 0
```

**Detach without stopping training:** `Ctrl+B` then `D`

**Re-attach to check progress:** `tmux attach -t spine_train`

**List sessions:** `tmux ls`

### 4.4 Monitor Training Progress

Open a **new SSH terminal** (do not interrupt the tmux session):

```bash
# GPU utilization — should be 90-100%
watch -n 5 nvidia-smi

# Live training log
tail -f "$nnUNet_results/Dataset202_TotalSpineSeg_XRay_FullSpine_AP_Lat/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_"*.txt

# Quick loss check
grep "train_loss" \
  "$nnUNet_results/Dataset202_TotalSpineSeg_XRay_FullSpine_AP_Lat/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_"*.txt \
  | tail -5
```

**What healthy training looks like:**
- `train_loss` starts around `0.7–0.9` and decreases steadily over epochs
- `val_loss` tracks train_loss (should not diverge wildly)
- GPU at 95%+ utilisation
- Each epoch takes a consistent amount of time

### 4.5 Estimated Timeline on A100

| Stage | Estimated Time |
| :--- | :--- |
| Fingerprint extraction | 5–15 min |
| Experiment planning | 2–5 min |
| Preprocessing (15k images) | 30–60 min |
| Training (1,000 epochs) | 24–48 hours |
| Test evaluation | 10–20 min |

---

## PHASE 5 — Post-Training Clinical Verification

### 5.1 Verify Final Checkpoint Exists

```bash
CHECKPOINT=$(find "$nnUNet_results" -name "checkpoint_final.pth" 2>/dev/null | head -1)

if [ -n "$CHECKPOINT" ]; then
    echo "✅ Final checkpoint found at:"
    echo "   $CHECKPOINT"
    ls -lh "$CHECKPOINT"
else
    echo "❌ No checkpoint_final.pth found."
    echo "Available checkpoints:"
    find "$nnUNet_results" -name "*.pth" 2>/dev/null
fi
```

### 5.2 Run Test Inference on One Image

```bash
TEST_IMG=$(ls "$nnUNet_raw/Dataset202_TotalSpineSeg_XRay_FullSpine_AP_Lat/imagesTs/" | head -1)
TEST_IMG_PATH="$nnUNet_raw/Dataset202_TotalSpineSeg_XRay_FullSpine_AP_Lat/imagesTs/$TEST_IMG"
OUTPUT_DIR="$TOTALSPINESEG_DATA/test_inference_output"

echo "Running inference on: $TEST_IMG"

totalspineseg_xray_inference \
  "$TEST_IMG_PATH" \
  "$OUTPUT_DIR" \
  --dataset-id 202 \
  --device cuda \
  --ordered-labels "C1-S1" \
  --overwrite
```

### 5.3 Verify Clinical Outputs

```bash
echo "=== Output Files ==="
ls -la "$OUTPUT_DIR/"

echo ""
echo "=== Clinical Metrics ==="
python3 << 'EOF'
import json, os

path = os.path.join(os.environ["TOTALSPINESEG_DATA"],
                    "test_inference_output", "postprocess_summary.json")
with open(path) as f:
    s = json.load(f)

case = s["cases"][0]
print(f"Components found:     {len(case['components'])}")

m = case.get("clinical_metrics", {})
print(f"Max Cobb Angle:       {m.get('max_cobb_angle', 'MISSING'):.2f}°")
print(f"Cobb vertebrae index: {m.get('cobb_vertebrae_indices', 'MISSING')}")

c0 = case["components"][0]
print(f"Surgical corners:     {'PRESENT' if c0.get('corners') else 'MISSING'}")
hr = c0.get("height_ratio")
print(f"Height ratio (VCF):   {hr:.3f}" if hr else "Height ratio: MISSING")
EOF
```

**Expected output:**
```
Components found:     18
Max Cobb Angle:       12.45°
Cobb vertebrae index: [2, 8]
Surgical corners:     PRESENT
Height ratio (VCF):   0.982
```

### 5.4 Download Preview Image for Visual Check

```bash
PREVIEW=$(ls "$OUTPUT_DIR/preview/" | head -1)
cp "$OUTPUT_DIR/preview/$PREVIEW" ~/test_preview.png
echo "Download with: scp root@<SERVER_IP>:~/test_preview.png ."
```

The preview image should show:
- ✅ Bone-shaped coloured overlays (NOT blurry rectangles)
- ✅ Each vertebra labelled by name (T1, L3, S1 etc.)
- ✅ Clear separation between each vertebra — no merged blobs

---

## TROUBLESHOOTING REFERENCE

### `ImportError: No module named 'totalspineseg'`
```bash
cd ~/TotalSpineSeg-Xray-Analysis
pip install -e .
# Or:
export PYTHONPATH="$HOME/TotalSpineSeg-Xray-Analysis:$PYTHONPATH"
```

### `FileNotFoundError: No compatible files were found`
```bash
# Check what extensions your images actually have
ls "$TOTALSPINESEG_DATA/xray/images/" | head -10

# If images are .jpg but pipeline expects .png:
find "$TOTALSPINESEG_DATA/xray/images/" -name "*.jpg" | while read f; do
  python3 -c "from PIL import Image; Image.open('$f').save('${f%.jpg}.png')"
  rm "$f"
done
```

### `CUDA Out of Memory` during training
```bash
# Pass a lower GPU memory target (in GB) as the 6th argument
bash scripts/train_xray.sh 202 0 nnUNetTrainer ExperimentPlanner nnUNetPlans 24
# Adjust the last number to your GPU's actual free VRAM
```

### `KeyError` on VinDr CSV column names
```bash
# Inspect actual column names
head -1 "$VINDR_CSV"
# Then pass explicit flags
python3 scripts/xray_landmarks_to_mask.py \
  --case-column image_id \
  --point-columns xl yt xr yt xr yb xl yb \
  ... (rest of flags)
```

### Image count ≠ Mask count
```bash
# Find the mismatched images
comm -23 \
  <(ls "$TOTALSPINESEG_DATA/xray/images/" | sed 's/\.png//' | sort) \
  <(ls "$TOTALSPINESEG_DATA/xray/masks/"  | sed 's/\.png//' | sort) \
  > /tmp/missing_masks.txt

echo "$(wc -l < /tmp/missing_masks.txt) images are missing masks:"
head -10 /tmp/missing_masks.txt

# Re-run the mask generation with --overwrite to regenerate all
```

### tmux session lost (SSH dropped during training)
```bash
# Check if nnUNet is still running
ps aux | grep nnUNetv2_train

# If yes — it's still alive, just disconnected from view
# Attach a new tmux session and tail the log:
tail -f "$nnUNet_results/Dataset202_TotalSpineSeg_XRay_FullSpine_AP_Lat/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_"*.txt
```

### Training loss not decreasing after 100 epochs
- Check GPU is at 90%+: `nvidia-smi`
- Check batch size isn't too small — look at `nnUNetPlans.json` in `nnUNet_preprocessed`
- Verify masks are not all-zero: `python3 -c "from PIL import Image; import numpy as np; img=Image.open('data/xray/masks/CASE.png'); print(np.asarray(img).max())"`

---

## FINAL SUMMARY — What "Done" Looks Like

After this entire workflow completes successfully, the system will:

1. **Ingest** any medical X-ray (PNG, JPG, DICOM `.dcm`) automatically.
2. **Segment** all vertebrae from C1 (skull base) to S1 (sacrum) with anatomical bone contours.
3. **Separate** merged thoracic vertebrae using Watershed mathematics.
4. **Label** each vertebra with its clinical name (C2, T4, L3, S1) on the output image.
5. **Calculate** Cobb's Angle for scoliosis assessment automatically.
6. **Detect** compression fractures via Anterior/Posterior height ratio.
7. **Export** surgical-grade mesh coordinates in JSON for external planners.

**Output generated for each X-ray processed:**

```
output/
├── binary/                    ← Clean binary foreground mask
├── ordered/                   ← Vertebrae numbered 1-N (superior to inferior)
├── labeled/                   ← Anatomically named vertebra mask
├── preview/                   ← Clinical overlay with professional text labels
├── input/                     ← Standardized copy of input image
├── raw/                       ← Raw nnUNet output (before post-processing)
└── postprocess_summary.json   ← Full clinical report with Cobb's angle + surgical JSON
```

---
*Guide version: 2026-04-28 | Commit: 2723a92 | Contact: engr.umairejaz@gmail.com*
