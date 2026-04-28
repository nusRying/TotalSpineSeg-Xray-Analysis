# TotalSpineSeg: Clinical X-Ray Diagnostic Suite (2D)
[![Commiters](https://img.shields.io/badge/contributors-8-blue.svg)](#contributors)
[![Version](https://img.shields.io/badge/Milestone-4.5--Complete-green.svg)](#milestones)

**TotalSpineSeg Clinical X-Ray Edition** is a specialized diagnostic engine re-engineered for automated vertebral segmentation and surgical metric extraction on **2D Radiographs (AP and Lateral)**. 

Originally branched from the world-class MRI research project by [NeuroPoly](https://github.com/neuropoly/totalspineseg), this repository has been transformed into a **Production-Grade X-Ray Suite** capable of surgical-ready diagnostics on standard clinical hardware.

---

## 💎 Primary Clinical Features (2D X-Ray)

Our core contribution focuses on moving spine AI from "boxes" to **"Clinical Diagnostics"**:

1.  **Clinical Geometry Engine:** 
    - Automated **Cobb's Angle** (Scoliosis Magnitude).
    - **A/P Height Ratios** (Fracture Grade Detection).
    - Precise **Surgical Corner Mapping** exported as Clinical JSON.
2.  **Anatomical Morphometry:** High-fidelity bone masks that follow true vertebral anatomy (not just rectangles).
3.  **A100 Large-Scale Training:** Integrated pipeline for **~15,500 images** (AASCE, VinDr, CSXA).
4.  **Hospital-Ready:** Native **DICOM** ingestion and post-processing for immediate pre-op planning.

---

## 🚀 Quick Start (X-Ray Suite)

### Diagnostic Inference
Process any clinical X-ray (PNG/JPG/DICOM) and get a full diagnostic report + anatomical overlays.
```bash
totalspineseg_xray_inference input_iamge.dcm output_dir/
```

### Post-Processing & Geometry
Run specifically for surgical metrics on an existing mask:
```bash
totalspineseg_xray_postprocess mask.png output_dir/ --geometry
```

### Server Deployment
Full blueprint for A100 GPU training and large-scale dataset acquisition:
👉 [**View Server Deployment Guide**](docs/server_deployment_guide.md)

---

## 🧠 Medical Metrics Overview
| Metric | Diagnostic Value |
| :--- | :--- |
| **Cobb's Angle** | Automated scoliosis curve measurement. |
| **Height Ratios** | Identification of Genant Grade vertebral collapses. |
| **JSON Export** | Cartesian coordinates for surgical navigation. |

---

## 🔧 Legacy 3D MRI Module (Original Core)

TotalSpineSeg remains compatible with its original mission: **3D MRI Instance Segmentation.** 

*For MRI-specific instructions, hybrid model descriptions, and NIfTI workflows, please refer to the [MRI Technical Documentation](docs/mri_legacy_guide.md) (previously located in the header).*

---

## 🤝 Contributors & Research Origin

This project is a hybrid of world-class academic research and production-grade clinical engineering:

- **X-Ray Suite Adaptation:** Re-engineered for surgical-grade 2D diagnostics (Umair Ejaz).
- **Core MRI Architecture:** Built by the [NeuroPoly Lab](https://github.com/neuropoly) (Molinier, Warszawer, Cohen-Adad).

---

## 📜 Citations

If you use the core nnU-Net architecture of this project, please cite:
> Isensee, F., et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
