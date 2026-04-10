# scripts

This folder contains bash and Python scripts used for preparing, training, and inference of the model.

Current MRI workflow:

- `prepare_datasets.sh`
- `train.sh`

Milestone 1 X-ray workflow:

- `xray_landmarks_to_mask.py`
- `prepare_xray_dataset.py`
- `train_xray.sh`
- `train_xray.ps1`

Current X-ray scope:

- AP thoracolumbar radiographs with `T1-L5` landmark coverage
- thoracic and lumbar vertebrae only unless a separate cervical dataset is added
- landmark rasterization can now emit anatomical vertebra labels such as `T1-L5` or `C1-L5` when the annotation order is known

See [README](../README.md) for the MRI pipeline, [docs/xray_milestone1_design.md](../docs/xray_milestone1_design.md) for the milestone 1 design, and [docs/xray_milestone2_workflow.md](../docs/xray_milestone2_workflow.md) for the milestone 2 inference and evaluation workflow.
