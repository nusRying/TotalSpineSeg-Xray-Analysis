# Project Context: TotalSpineSeg X-ray Analysis

## End Goal
Develop an automated pipeline for:
1. Identifying spinal regions in full-body X-rays.
2. Generating precise anatomical segmentation masks for vertebrae (C1-C7, T1-T12, L1-L5).
3. Calculating the Cobb Angle automatically based on the segmented masks.

## Current Infrastructure
- **Local Environment:** Windows PC (`c:\Users\umair\Videos\Freelance\Sunshine V2`). Used for scripts and small tests.
- **Remote Environment:** A100 GPU Server (`154.54.102.35:10902`, user: `root`). All heavy datasets (100GB+) and training will occur here.
- **Access Script:** `scratch/run_remote_tool.py` and `filename` (private key) are used locally to execute SSH/SFTP commands on the remote server.

## Dataset Inventory (On Remote Server)
| Dataset | Type | Region | Labels | Status | Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mendeley Lumbar** | X-ray | L1-L5 | Full Anatomical Masks | Ready (Converted to nnU-Net format) | `/workspace/nnUNet_raw/Dataset250_MendeleyLumbar/` |
| **CSXA_V3** | X-ray | C1-C7 | Landmarks (Dots) | Ready | `/workspace/data/raw_external/CSXA_V3/` |
| **VinDr-SpineXR** | X-ray | Full Spine | Bounding Boxes (Pathology) | Ready | `/workspace/data/raw_external/VinDr-SpineXR/` |
| **CTSpine1K** | CT Scan | Full Spine | 3D NIfTI Masks | ⬇️ Downloading (~150GB) | `/workspace/data/raw_external/CTSpine1K/` |

## Current Progress & Immediate State
1. **Mendeley Lumbar** (1,200 images) was successfully downloaded and converted into the `nnUNet_raw` format. It is completely ready for training the L1-L5 segmentation model.
2. **LumASe** (663 CT scans of individual L1-L5 vertebrae) was downloaded.
3. We uploaded `generate_drr.py` and `utils.py` to the remote `/workspace/` to convert the 3D NIfTI scans into 2D synthetic X-rays (DRRs).
4. **Current Status of DRR Generation:** The dependencies (`nibabel`, `numpy`, `Pillow`) were successfully installed directly into the remote `/workspace/`. A test run on 5 LumASe cases successfully produced synthetic X-rays using `process_lumase.py`.
5. **Thoracic Gap Solved:** We identified that the **CTSpine1K** dataset on Hugging Face provides full spine (Cervical, Thoracic, Lumbar) 3D CT masks. A background script (`nohup python3 /workspace/download_ctspine.py`) is currently running on the server to download the 150GB raw NIfTI format.

## What Needs to Happen Next
1. **Monitor CTSpine1K Download:** Periodically check `/workspace/ctspine_download.log` on the server to track the 150GB download progress.
2. **Execute Full DRR Generation:** Update `process_lumase.py` (and potentially create a similar script for CTSpine1K) to run on all available CT cases, generating a massive 2D X-ray mask dataset for the entire spine.
3. **Train Models:** Start training the nnU-Net models on the prepared datasets.

