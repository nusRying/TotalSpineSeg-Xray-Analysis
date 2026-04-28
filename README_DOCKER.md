# TotalSpineSeg X-Ray Docker Guide

This guide explains how to build and run the 2D X-ray spine segmentation pipeline using Docker.

## 1. Build the Image
From the project root directory, run:
```bash
docker build -t totalspineseg-xray .
```

## 2. Run Inference
To run inference on your X-ray images, use the following command. You will need to mount your local folders for input images and output results.

```bash
docker run --gpus all \
    -v /path/to/your/images:/images \
    -v /path/to/save/results:/output \
    -v /path/to/nnUNet_results:/app/data/nnUNet/results \
    totalspineseg-xray /images /output --device cuda
```

### Parameters explained:
*   `--gpus all`: Enables GPU acceleration (required for A100/EC2).
*   `-v /path/to/your/images:/images`: Mounts your local X-ray folder to `/images` inside the container.
*   `-v /path/to/save/results:/output`: Mounts your local output folder to `/output`.
*   `-v /path/to/nnUNet_results:/app/data/nnUNet/results`: Mounts the folder containing the `Dataset202` model weights.

## 3. Example Command (AP/Lateral Numbering)
If you want to perform automatic numbering (e.g., C1-L5), add the `--ordered-labels` flag:

```bash
docker run --gpus all \
    -v /home/ubuntu/data/xrays:/images \
    -v /home/ubuntu/results:/output \
    -v /home/ubuntu/weights:/app/data/nnUNet/results \
    totalspineseg-xray /images /output --device cuda --ordered-labels "C1-L5"
```
