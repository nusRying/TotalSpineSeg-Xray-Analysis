# TotalSpineSeg X-Ray: Milestone 4 Production Guide

This guide explains how to build and run the 2D X-ray spine segmentation pipeline. This version is **self-contained**, meaning the model weights are built directly into the container or searched in the internal `weights/` folder.

## 1. Build the Production Image
From this directory, build the official production container:
```bash
docker build -t totalspineseg-production .
```

## 2. Quick Test (Using Samples)
We have provided 10 sample X-rays in the `samples/` folder. You can test the entire pipeline with a single command:
```bash
docker run --gpus all -v "$(pwd):/app" totalspineseg-production samples sample_results
```
*After running, check the `sample_results/preview/` folder to see the segmentation overlays.*

## 3. Run on Your Own Images
To process a folder of clinical images on a GPU-enabled server:
```bash
docker run --gpus all -v "$(pwd):/app" \
    -v "/path/to/your/xrays:/input_images" \
    totalspineseg-production /input_images /app/my_results --device cuda
```

## 4. Run on a Single Image
You can also run the model on a specific file rather than a whole folder:
```bash
docker run --gpus all -v "$(pwd):/app" totalspineseg-production \
    samples/0001035_0000.png \
    single_result --device cuda
```

## 5. Advanced Features
### Anatomical Labeling (Vertebrae Numbering)
To enable automated anatomical identification (e.g., T1 through L5), add the `--ordered-labels` flag:
```bash
docker run --gpus all -v "$(pwd):/app" totalspineseg-production \
    samples labeling_results --ordered-labels "T1-L5"
```

### Output Formats
The system generates:
1. **`preview/`**: High-resolution PNGs with colored segmentation overlays.
2. **`binary/`**: Black and white masks of the entire spine.
3. **`ordered/`**: Mask where each vertebra has a unique integer ID for analysis.
4. **`inference_manifest.json`**: Metadata and validation stats for the run.
