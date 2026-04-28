# TotalSpineSeg: 3D MRI Module (Legacy Core)

TotalSpineSeg is a tool for automatic instance segmentation of all vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal in MRI images. It is robust to various MRI contrasts, acquisition orientations, and resolutions.

## Model Description

TotalSpineSeg uses an hybrid approach that integrates nnU-Net with an iterative algorithm for instance segmentation and labeling of vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal.

**Step 1**: An nnU-Net model (`Dataset101`) is used to identify nine classes in total:
- Four semantic classes: spinal cord, spinal canal, IVDs, and vertebrae.
- Five landmark classes: C2-C3, C7-T1, T12-L1, and L5-S.

**Step 2:** A second nnU-Net model (`Dataset102`) assigns individual labels to each vertebra and IVD.

## Datasets (MRI)
- [whole-spine](https://openneuro.org/datasets/ds005616/versions/1.0.0)
- [SPIDER](https://doi.org/10.5281/zenodo.10159290)
- [Spine Generic Project](https://github.com/spine-generic)

## Installation (MRI-Optimized)
```bash
git clone https://github.com/neuropoly/totalspineseg.git
python3 -m pip install -e totalspineseg[nnunetv2]
```

## Inference (MRI)
```bash
totalspineseg INPUT OUTPUT_FOLDER [--step1] [--iso]
```

## How to cite us (MRI Research)
```
@article{warszawer2025totalspineseg,
   title={TotalSpineSeg: Robust Spine Segmentation with Landmark-Based Labeling in MRI},
   author={Warszawer, Yehuda et al.},
   year={2025},
   journal={ResearchGate preprint}
}
```
