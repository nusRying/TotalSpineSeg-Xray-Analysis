# TotalSpineSeg X-Ray: Technical Implementation Guides

This document contains the engineering blueprints for Milestone 4 (High-Accuracy Training) and Milestone 6 (Metrics Extraction).

---

## **Part 1: A100 High-Accuracy Training Guide (Milestone 4)**
*Goal: Leverage A100 VRAM to fix scale issues and reach 95%+ precision required for medical metrics.*

### **1. Accuracy Improvement Strategy**
To transition from 91% to 95%+ Dice and ensure medical-grade metrics, we utilize the A100's high VRAM (40GB/80GB) for three key optimizations:

*   **A. High-Fidelity Resolution (1024px):** 
    *   *Problem:* Standard training often downsamples to 512px, losing the sharp corners of vertebrae.
    *   *Fix:* Force a **1024x1024 internal patch size**. This allows the AI to see the exact edge of the vertebral endplates, which is critical for accurate Cobb Angle and Listhesis math.
*   **B. Gradient Stability (Batch Size 16-32):**
    *   *Problem:* Low batch sizes (2-4) lead to noisy gradients and "jagged" segmentation edges.
    *   *Fix:* Increase batch size to **16 or 32**. This stabilizes the training process and produces significantly smoother, more anatomically correct masks.
*   **C. Extended Convergence (2000 Epochs):**
    *   *Strategy:* Use the A100's speed to double the training iterations. This ensures the model fully learns the variation in synthetic phantom data vs. real clinical data.

### **2. Plan Optimization (Manual Tweak)**
After running the initial preprocessing, we must optimize the `nnUNetPlans.json` to leverage the A100:
1.  **Locate:** `data/nnUNet/preprocessed/Dataset202.../nnUNetPlans.json`
2.  **Modify:** 
    *   `patch_size`: Set to `[1024, 1024]`
    *   `batch_size`: Set to `16` (or higher depending on VRAM usage)
    *   `n_conv_per_stage_encoder`: (Optional) Increase for deeper feature extraction.

### **3. AWS Execution Workflow**
1.  **Build:** `docker build -t totalspineseg-a100 .`
2.  **Dataset Injection:** 
    ```bash
    docker run --rm -v /home/ubuntu/data:/data totalspineseg-a100 \
    python scripts/prepare_xray_dataset.py --images-dir /data/imgs --labels-dir /data/masks --output-root /app/data/nnUNet/raw --dataset-id 202
    ```
3.  **Run Optimized Training:**
    ```bash
    # Use --npz to save probabilities for high-res metric extraction later
    docker run --gpus all -v /home/ubuntu/SunshineV2:/app totalspineseg-a100 \
    nnUNetv2_train 202 2d 0 --npz
    ```

### **4. Monitoring "Learning Stats" for the Client**
*   **Loss Curves:** Monitor `progress.png` in the results folder. Look for a smooth, exponential decay in `train_loss`.
*   **Metric Success:** Provide the client with the final `val_dice` score from the `training_log.txt`. On an A100 with 1024px resolution, we target a **vertebrae Dice > 0.94**.

---

## **Part 2: Spine Metrics Geometry Engine (Milestone 6)**
*Goal: Algorithmic extraction of clinical data from masks.*

### **1. Core Algorithm: Vertebral Corner Detection**
All metrics depend on identifying the 4 POIs (Points of Interest) for each vertebra:
*   **SP/SA:** Superior-Posterior / Superior-Anterior
*   **IP/IA:** Inferior-Posterior / Inferior-Anterior

### **2. Clinical Calculations**
*   **Listhesis (Slip):** `abs(V_top.IP.x - V_bottom.SP.x)`
*   **Disc Height:** Vertical "Clear Space" between `V_top.Inferior_Endplate` and `V_bottom.Superior_Endplate`.
*   **Cobb Angle:** Calculated as the maximum divergence angle between any two endplate slopes ($\theta = \arctan(m)$).
*   **Height Loss %:** Current vertebral height vs. the average height of adjacent vertebrae.
