# TotalSpineSeg X-Ray Milestone 2 Workflow

## Scope

Milestone 2 turns the milestone 1 X-ray scaffold into a usable baseline workflow:

- inference wrapper for 2D X-ray images
- connected-component cleanup
- superior-to-inferior vertebra ordering
- binary evaluation reports
- preview overlays for quick visual review

This remains an experimental X-ray path. It does not claim parity with the MRI pipeline.

If the selected public dataset is labeled from `T1-L5`, the trained baseline will cover the thoracic and lumbar spine only. Cervical coverage requires additional `C1-C7` annotations from another dataset.
The code path now supports either binary vertebra cleanup or labeled vertebra outputs when the model prediction or ordered annotation scheme provides anatomical labels such as `T1-L5` or `C1-L5`.

## Commands

Inference and postprocessing:

```bash
totalspineseg_xray_inference INPUT OUTPUT --dataset-id 201 --device cuda
```

If you already have raw predicted masks and only want postprocessing:

```bash
totalspineseg_xray_inference INPUT OUTPUT --raw-preds-dir RAW_PREDS
```

Standalone postprocessing:

```bash
totalspineseg_xray_postprocess RAW_PREDS OUTPUT --image-dir INPUT --min-area 128
```

Evaluation:

```bash
totalspineseg_xray_evaluate PRED_BINARY LABELS REPORT
```

## Inference Output Layout

`totalspineseg_xray_inference` writes:

- `input/`: staged grayscale `.png` images named for nnU-Net
- `raw/`: raw model predictions
- `binary/`: cleaned binary vertebra masks
- `ordered/`: sequential vertebra component labels ordered superior-to-inferior
- `labeled/`: anatomical vertebra labels when available or assigned during postprocessing
- `preview/`: overlay previews for quick inspection
- `inference_manifest.json`: run configuration
- `postprocess_summary.json`: per-case component summaries

## Evaluation Outputs

`totalspineseg_xray_evaluate` writes:

- `per_case_metrics.csv`
- `metrics_summary.json`

Current metrics:

- Dice
- IoU
- Precision
- Recall
- component counts
- foreground area statistics
- optional multiclass label Dice / IoU / Precision / Recall when evaluating labeled masks

## Recommended First Real-Data Run

1. Prepare thoracolumbar landmark-derived masks with `scripts/xray_landmarks_to_mask.py`.
2. Build the nnU-Net dataset with `scripts/prepare_xray_dataset.py`.
3. Train the 2D baseline with `scripts/train_xray.sh`.
4. Run `totalspineseg_xray_inference` on the held-out split.
5. Run `totalspineseg_xray_evaluate` against held-out labels.

## Remaining Dataset-Dependent Work

The repository now has the milestone 2 code path, but these still depend on the actual public dataset:

- real annotation export into the canonical CSV schema
- real baseline training run on the selected dataset
- held-out validation metrics on real X-ray cases
- tuning of postprocessing thresholds such as `min-area`
