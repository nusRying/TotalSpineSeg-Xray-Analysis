"""
TotalSpineSeg X-Ray: Production Inference Pipeline
Version: 2026.04.25 (Milestone 4)

This script manages the end-to-end inference workflow for 2D spine X-rays, including:
1. Adaptive Resolution Scaling (Standardizing to 1024px)
2. GPU-Accelerated nnU-Net Prediction
3. Automated Post-processing and Resolution Restoration
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image

from .common import (
    SUPPORTED_IMAGE_SUFFIXES,
    SUPPORTED_LABEL_SUFFIXES,
    case_id_from_path,
    collect_case_files,
    load_label_image,
    save_label_image,
    write_json,
)
from .labels import ordered_label_values_from_spec
from .postprocess import postprocess_folder


def repo_nnunet_dirs() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    nnunet_root = repo_root / "data" / "nnUNet"
    
    # Priority: If an integrated 'weights' folder exists, use it for results
    results_dir = repo_root / "weights"
    if not results_dir.exists():
        results_dir = nnunet_root / "results"
        
    return {
        "nnUNet_raw": str(nnunet_root / "raw"),
        "nnUNet_preprocessed": str(nnunet_root / "preprocessed"),
        "nnUNet_results": str(results_dir),
    }


def staged_case_id(path: Path) -> str:
    case_id = case_id_from_path(path)
    if case_id.endswith("_0000"):
        return case_id[:-5]
    return case_id


def calculate_scale(width: int, height: int, target_size: int) -> float:
    return float(target_size) / max(width, height)


def stage_single_image(input_path: Path, output_path: Path, target_size: int | None = None) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(input_path) as image:
            orig_width, orig_height = image.size
            if target_size is not None:
                scale = calculate_scale(orig_width, orig_height, target_size)
                if scale < 1.0:
                    new_width = int(round(orig_width * scale))
                    new_height = int(round(orig_height * scale))
                    image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            image.convert("L").save(output_path)
            return orig_width, orig_height
    except Exception as e:
        print(f"❌ ERROR: Failed to process image {input_path}. Details: {e}")
        return 0, 0


def stage_input_images(input_path: Path, staged_dir: Path, target_size: int | None = None, overwrite: bool = False) -> tuple[dict[str, Path], dict[str, tuple[int, int]]]:
    if input_path.is_file():
        case_id = staged_case_id(input_path)
        staged_path = staged_dir / f"{case_id}_0000.png"
        shape = (0, 0)
        if overwrite or not staged_path.exists():
            shape = stage_single_image(input_path, staged_path, target_size=target_size)
        else:
            with Image.open(input_path) as image:
                shape = image.size
        return {case_id: staged_path}, {case_id: shape}

    source_images = collect_case_files(input_path, SUPPORTED_IMAGE_SUFFIXES)
    staged = {}
    shapes = {}
    for _, source_path in sorted(source_images.items()):
        case_id = staged_case_id(source_path)
        staged_path = staged_dir / f"{case_id}_0000.png"
        if overwrite or not staged_path.exists():
            shape = stage_single_image(source_path, staged_path, target_size=target_size)
        else:
            with Image.open(source_path) as image:
                shape = image.size
        staged[case_id] = staged_path
        shapes[case_id] = shape
    return staged, shapes


def copy_raw_predictions(raw_predictions_dir: Path, output_dir: Path, overwrite: bool = False) -> dict[str, Path]:
    predictions = collect_case_files(raw_predictions_dir, SUPPORTED_LABEL_SUFFIXES)
    copied = {}
    for case_id, source_path in sorted(predictions.items()):
        destination_path = output_dir / f"{case_id}.png"
        if overwrite or not destination_path.exists():
            save_label_image(load_label_image(source_path), destination_path)
        copied[case_id] = destination_path
    return copied


def run_nnunet_predict(
    staged_input_dir: Path,
    raw_output_dir: Path,
    dataset_id: int,
    configuration: str,
    trainer: str,
    plans: str,
    folds: list[str],
    device: str,
    checkpoint: str,
):
    if os.name == "nt":
        env_executable = Path(sys.executable).parent / "Scripts" / "nnUNetv2_predict.exe"
    else:
        env_executable = Path(sys.executable).with_name("nnUNetv2_predict")
    executable = str(env_executable) if env_executable.exists() else shutil.which("nnUNetv2_predict")
    if executable is None:
        raise FileNotFoundError("Could not locate nnUNetv2_predict in the active environment.")
    env = os.environ.copy()
    for key, value in repo_nnunet_dirs().items():
        if Path(value).exists():
            env[key] = value
    command = [
        executable,
        "-d",
        str(dataset_id),
        "-i",
        str(staged_input_dir),
        "-o",
        str(raw_output_dir),
        "-c",
        configuration,
        "-tr",
        trainer,
        "-p",
        plans,
        "-chk",
        checkpoint,
        "-device",
        device,
        "-f",
        *folds,
    ]
    subprocess.run(command, check=True, env=env)


def run_xray_inference(
    input_path: Path,
    output_path: Path,
    dataset_id: int = 202,
    configuration: str = "2d",
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
    folds: list[str] | None = None,
    device: str = "cpu",
    checkpoint: str = "checkpoint_final.pth",
    raw_predictions_dir: Path | None = None,
    target_size: int | None = 1024,
    min_area: int = 64,
    max_components: int | None = None,
    max_components_per_label: int | None = 1,
    prediction_mode: str = "auto",
    ordered_label_values: list[int] | None = None,
    ordered_label_anchor: str = "strict",
    label_map_json: Path | None = None,
    overwrite: bool = False,
):
    if folds is None:
        folds = ["0"]

    # Pre-flight check: Verify weights exist if not using raw_predictions_dir
    if raw_predictions_dir is None:
        dirs = repo_nnunet_dirs()
        weights_dir = Path(dirs["nnUNet_results"]) / f"Dataset{dataset_id:03d}_TotalSpineSeg_XRay_FullSpine_AP_Lat" / f"{trainer}__{plans}__{configuration}"
        if not weights_dir.exists():
            print(f"\n⚠️  WARNING: Could not find weights at {weights_dir}")
            print(f"Please ensure the 'nnUNet_results' environment variable is set correctly.")
            print(f"Current search path: {dirs['nnUNet_results']}\n")
            # We don't raise error here yet as user might have a different naming convention, 
            # but we warn them.

    output_path.mkdir(parents=True, exist_ok=True)
    staged_input_dir = output_path / "input"
    raw_output_dir = output_path / "raw"

    staged_inputs, staged_shapes = stage_input_images(input_path, staged_input_dir, target_size=target_size, overwrite=overwrite)

    if raw_predictions_dir is not None:
        copy_raw_predictions(raw_predictions_dir, raw_output_dir, overwrite=overwrite)
    else:
        run_nnunet_predict(
            staged_input_dir=staged_input_dir,
            raw_output_dir=raw_output_dir,
            dataset_id=dataset_id,
            configuration=configuration,
            trainer=trainer,
            plans=plans,
            folds=folds,
            device=device,
            checkpoint=checkpoint,
        )

    # Use original images for post-processing overlays to match resampled mask dimensions
    original_image_dir = input_path if input_path.is_dir() else input_path.parent
    
    # Determine the suffix to use based on the original input filename(s)
    image_suffix_to_use = "_0000" if (
        input_path.is_dir() or 
        input_path.stem.endswith("_0000") or 
        input_path.name.endswith("_0000.nii.gz")
    ) else ""

    postprocess_summary = postprocess_folder(
        input_dir=raw_output_dir,
        output_dir=output_path,
        image_dir=original_image_dir,
        image_suffix=image_suffix_to_use,
        staged_shapes=staged_shapes,
        min_area=min_area,
        max_components=max_components,
        max_components_per_label=max_components_per_label,
        prediction_mode=prediction_mode,
        ordered_label_values=ordered_label_values,
        ordered_label_anchor=ordered_label_anchor,
        label_map_json=label_map_json,
        overwrite=overwrite,
    )
    manifest = {
        "dataset_id": int(dataset_id),
        "configuration": configuration,
        "trainer": trainer,
        "plans": plans,
        "folds": folds,
        "device": device,
        "checkpoint": checkpoint,
        "num_inputs": len(staged_inputs),
        "target_size": target_size,
        "staged_shapes": {k: list(v) for k, v in staged_shapes.items()},
        "used_raw_predictions_dir": None if raw_predictions_dir is None else str(raw_predictions_dir),
        "prediction_mode": prediction_mode,
        "ordered_label_values": ordered_label_values,
        "ordered_label_anchor": ordered_label_anchor,
        "label_map_json": None if label_map_json is None else str(label_map_json),
    }
    write_json(manifest, output_path / "inference_manifest.json")
    return postprocess_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run the Full-Spine (C1-S1) X-ray nnU-Net model and postprocess vertebra components.",
        epilog=textwrap.dedent(
            """\
            Examples:
              totalspineseg_xray_inference images out --dataset-id 202 --device cuda --ordered-labels "C1-L5"
              totalspineseg_xray_inference images out --ordered-labels "T1-L5"
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input X-ray image or folder of images.")
    parser.add_argument("output", type=Path, help="Output folder.")
    parser.add_argument("--dataset-id", type=int, default=202, help="nnU-Net dataset id.")
    parser.add_argument("--configuration", default="2d", help="nnU-Net configuration name.")
    parser.add_argument("--trainer", default="nnUNetTrainer", help="nnU-Net trainer name.")
    parser.add_argument("--plans", default="nnUNetPlans", help="nnU-Net plans name.")
    parser.add_argument("--folds", nargs="+", default=["0"], help="Fold ids passed to nnUNetv2_predict.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_final.pth",
        help="Checkpoint filename passed to nnUNetv2_predict, e.g. checkpoint_best.pth.",
    )
    parser.add_argument(
        "--raw-preds-dir",
        type=Path,
        default=None,
        help="Skip nnU-Net prediction and use an existing folder of raw predicted masks.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1024,
        help="Standardize input image long-edge to this size (adaptive scaling). Set to 0 to disable.",
    )
    parser.add_argument("--min-area", type=int, default=64, help="Minimum area for connected vertebra components.")
    parser.add_argument("--max-components", type=int, default=None, help="Optional cap on kept components in binary mode.")
    parser.add_argument(
        "--max-components-per-label",
        type=int,
        default=1,
        help="Optional cap on kept connected components for each class in multiclass mode.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["auto", "binary", "multiclass"],
        default="auto",
        help="How to interpret the raw model output before postprocessing.",
    )
    parser.add_argument(
        "--ordered-labels",
        default="",
        help='Optional superior-to-inferior anatomical label specification such as "T1-L5" or "C1-L5".',
    )
    parser.add_argument(
        "--ordered-label-anchor",
        choices=["strict", "superior", "inferior"],
        default="strict",
        help="How to align ordered anatomical labels with the detected vertebra components.",
    )
    parser.add_argument(
        "--label-map-json",
        type=Path,
        default=None,
        help="Optional JSON file that maps anatomical label names to integer ids.",
    )
    parser.add_argument("--overwrite", "-r", action="store_true", default=False)
    args = parser.parse_args()

    ordered_label_values = None
    if args.ordered_labels:
        ordered_label_values = ordered_label_values_from_spec(args.ordered_labels, args.label_map_json)

    run_xray_inference(
        input_path=args.input,
        output_path=args.output,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        trainer=args.trainer,
        plans=args.plans,
        folds=args.folds,
        device=args.device,
        checkpoint=args.checkpoint,
        raw_predictions_dir=args.raw_preds_dir,
        min_area=args.min_area,
        max_components=args.max_components,
        max_components_per_label=args.max_components_per_label,
        prediction_mode=args.prediction_mode,
        ordered_label_values=ordered_label_values,
        ordered_label_anchor=args.ordered_label_anchor,
        label_map_json=args.label_map_json,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
