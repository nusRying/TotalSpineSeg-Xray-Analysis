import argparse
import json
import random
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from totalspineseg.xray.labels import infer_named_labels


SUPPORTED_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
SUPPORTED_LABEL_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 2D spine X-ray images and masks into an nnU-Net v2 raw dataset.",
        epilog=textwrap.dedent(
            """\
            Example:
              python scripts/prepare_xray_dataset.py ^
                --images-dir data/xray/images ^
                --labels-dir data/xray/masks ^
                --output-root data/nnUNet/raw ^
                --dataset-id 201 ^
                --dataset-name TotalSpineSeg_XRay_Thoracolumbar_AP
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder with source X-ray images.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Folder with source segmentation masks.")
    parser.add_argument("--output-root", type=Path, required=True, help="nnU-Net raw root directory.")
    parser.add_argument("--dataset-id", type=int, default=201, help="Target nnU-Net dataset id.")
    parser.add_argument(
        "--dataset-name",
        default="TotalSpineSeg_XRay_Thoracolumbar_AP",
        help="Target nnU-Net dataset name without the DatasetXXX_ prefix.",
    )
    parser.add_argument(
        "--file-ending",
        choices=[".png", ".bmp", ".tif"],
        default=".png",
        help="Lossless output format for images and labels.",
    )
    parser.add_argument("--image-suffix", default="", help="Suffix to strip from the image stem when matching cases.")
    parser.add_argument("--label-suffix", default="", help="Suffix to strip from the label stem when matching cases.")
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Optional JSON file with explicit train/test case ids.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of cases assigned to the test split when split-json is omitted.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic splitting.")
    parser.add_argument(
        "--channel-name",
        default="XRay",
        help="Channel name written into dataset.json.",
    )
    parser.add_argument(
        "--foreground-label-name",
        default="vertebrae",
        help="Foreground class name written into dataset.json for binary masks.",
    )
    parser.add_argument(
        "--label-map-json",
        type=Path,
        default=None,
        help="Optional JSON file that maps integer mask values to anatomical class names.",
    )
    parser.add_argument(
        "--reference",
        default="AASCE 2019 / SpineWeb Dataset 16 AP thoracolumbar radiographs with T1-L5 landmarks",
        help="Reference string written into dataset.json.",
    )
    parser.add_argument(
        "--description",
        default="2D AP thoracolumbar X-ray vertebrae segmentation baseline for TotalSpineSeg.",
        help="Description written into dataset.json.",
    )
    parser.add_argument(
        "--binarize-labels",
        action="store_true",
        default=False,
        help="Convert every positive label value to 1.",
    )
    parser.add_argument(
        "--skip-test-labels",
        action="store_true",
        default=False,
        help="Do not copy test labels into labelsTs.",
    )
    parser.add_argument(
        "--overwrite",
        "-r",
        action="store_true",
        default=False,
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def strip_case_id(path: Path, suffix_to_strip: str) -> str:
    stem = path.stem
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def collect_files(folder: Path, suffix_to_strip: str, allowed_suffixes: set[str]) -> dict[str, Path]:
    mapping = {}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        case_id = strip_case_id(path, suffix_to_strip)
        if case_id in mapping:
            raise ValueError(f'Duplicate case id "{case_id}" found under {folder}.')
        mapping[case_id] = path
    if not mapping:
        raise FileNotFoundError(f"No compatible files were found under {folder}.")
    return mapping


def load_explicit_split(split_json: Path) -> tuple[list[str], list[str]]:
    content = json.loads(split_json.read_text(encoding="utf-8"))
    train_ids = content.get("train") or content.get("training")
    test_ids = content.get("test") or content.get("testing")
    if not train_ids or test_ids is None:
        raise ValueError("split-json must contain train/training and test/testing lists.")
    return [str(case_id) for case_id in train_ids], [str(case_id) for case_id in test_ids]


def make_split(case_ids: list[str], test_size: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0 < test_size < 1:
        raise ValueError("test-size must be between 0 and 1.")
    if len(case_ids) < 2:
        raise ValueError("At least two matched cases are required to create a train/test split.")

    shuffled = case_ids[:]
    random.Random(seed).shuffle(shuffled)
    num_test = max(1, int(round(len(shuffled) * test_size)))
    num_test = min(num_test, len(shuffled) - 1)
    test_ids = sorted(shuffled[:num_test])
    train_ids = sorted(shuffled[num_test:])
    return train_ids, test_ids


def save_image(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        image.convert("L").save(output_path)


def save_label(input_path: Path, output_path: Path, overwrite: bool, binarize: bool) -> set[int]:
    if output_path.exists() and not overwrite:
        with Image.open(output_path) as image:
            return set(np.unique(np.asarray(image)).tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        label_array = np.asarray(image)

    if label_array.ndim != 2:
        raise ValueError(f"Labels must be single-channel integer masks. Problem file: {input_path}")

    if binarize:
        label_array = (label_array > 0).astype(np.uint8)
    else:
        if not np.issubdtype(label_array.dtype, np.integer):
            raise ValueError(f"Labels must be integer-valued. Problem file: {input_path}")

    Image.fromarray(label_array).save(output_path)
    return set(np.unique(label_array).tolist())


def write_dataset_json(
    dataset_dir: Path,
    channel_name: str,
    labels: dict[str, int],
    num_training: int,
    file_ending: str,
    dataset_name: str,
    reference: str,
    description: str,
) -> None:
    dataset_json = {
        "channel_names": {"0": channel_name},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": file_ending,
        "name": dataset_name,
        "reference": reference,
        "description": description,
    }
    dataset_dir.joinpath("dataset.json").write_text(
        json.dumps(dataset_json, indent=4) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()

    images = collect_files(args.images_dir, args.image_suffix, SUPPORTED_IMAGE_SUFFIXES)
    labels = collect_files(args.labels_dir, args.label_suffix, SUPPORTED_LABEL_SUFFIXES)
    common_case_ids = sorted(set(images) & set(labels))
    if not common_case_ids:
        raise ValueError("No matching image/label pairs were found.")

    if args.split_json is not None:
        train_ids, test_ids = load_explicit_split(args.split_json)
    else:
        train_ids, test_ids = make_split(common_case_ids, args.test_size, args.seed)

    missing = sorted(set(train_ids + test_ids) - set(common_case_ids))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Split references missing cases. First cases: {preview}")

    dataset_dir = args.output_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_dir / "imagesTr"
    images_ts = dataset_dir / "imagesTs"
    labels_tr = dataset_dir / "labelsTr"
    labels_ts = dataset_dir / "labelsTs"

    train_label_values: set[int] = set()

    for case_id in train_ids:
        save_image(images[case_id], images_tr / f"{case_id}_0000{args.file_ending}", args.overwrite)
        train_label_values.update(
            save_label(
                labels[case_id],
                labels_tr / f"{case_id}{args.file_ending}",
                args.overwrite,
                args.binarize_labels,
            )
        )

    for case_id in test_ids:
        save_image(images[case_id], images_ts / f"{case_id}_0000{args.file_ending}", args.overwrite)
        if not args.skip_test_labels:
            save_label(
                labels[case_id],
                labels_ts / f"{case_id}{args.file_ending}",
                args.overwrite,
                args.binarize_labels,
            )

    labels_dict = infer_named_labels(
        train_label_values,
        foreground_name=args.foreground_label_name,
        label_map_json=args.label_map_json,
    )
    write_dataset_json(
        dataset_dir=dataset_dir,
        channel_name=args.channel_name,
        labels=labels_dict,
        num_training=len(train_ids),
        file_ending=args.file_ending,
        dataset_name=f"Dataset{args.dataset_id:03d}_{args.dataset_name}",
        reference=args.reference,
        description=args.description,
    )

    split_manifest = {
        "train": train_ids,
        "test": test_ids,
        "file_ending": args.file_ending,
        "binarize_labels": args.binarize_labels,
        "label_map_json": None if args.label_map_json is None else str(args.label_map_json),
    }
    dataset_dir.joinpath("xray_split.json").write_text(
        json.dumps(split_manifest, indent=4) + "\n",
        encoding="utf-8",
    )

    print(
        f"Prepared {dataset_dir} with {len(train_ids)} training cases and {len(test_ids)} test cases. "
        f"Output format: {args.file_ending}"
    )


if __name__ == "__main__":
    main()
