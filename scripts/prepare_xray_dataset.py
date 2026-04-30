import argparse
import json
import random
import shutil
import sys
import textwrap
from pathlib import Path
from multiprocessing import Pool

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
        description=textwrap.dedent(
            """
            Prepare a TotalSpineSeg X-ray dataset for nnU-Net v2.
            
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
        help="Do not copy test labels to labelsTs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of background processes to use for conversion.",
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
    data = json.loads(split_json.read_text(encoding="utf-8"))
    return data["train"], data["test"]


def make_split(case_ids: list[str], test_size: float, seed: int) -> tuple[list[str], list[str]]:
    random.seed(seed)
    shuffled = list(case_ids)
    random.shuffle(shuffled)
    num_test = int(len(shuffled) * test_size)
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
        if binarize:
            label_array = (label_array > 0).astype(np.uint8)
        Image.fromarray(label_array).save(output_path)
        return set(np.unique(label_array).tolist())


def process_case(args_tuple):
    case_id, input_image, output_image, input_label, output_label, overwrite, binarize = args_tuple
    save_image(input_image, output_image, overwrite)
    label_values = save_label(input_label, output_label, overwrite, binarize)
    return label_values


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

    # Prepare tasks for Pool
    tasks = []
    for case_id in train_ids:
        tasks.append((
            case_id,
            images[case_id],
            images_tr / f"{case_id}_0000{args.file_ending}",
            labels[case_id],
            labels_tr / f"{case_id}{args.file_ending}",
            args.overwrite,
            args.binarize_labels
        ))

    print(f"Processing {len(train_ids)} training cases using {args.num_processes} processes...")
    with Pool(args.num_processes) as p:
        results = p.map(process_case, tasks)

    train_label_values = set()
    for res in results:
        train_label_values.update(res)

    print(f"Processing {len(test_ids)} test cases...")
    for case_id in test_ids:
        save_image(images[case_id], images_ts / f"{case_id}_0000{args.file_ending}", args.overwrite)
        if not args.skip_test_labels:
            save_label(
                labels[case_id],
                labels_ts / f"{case_id}{args.file_ending}",
                args.overwrite,
                args.binarize_labels,
            )

    if args.label_map_json is not None:
        label_names = json.loads(args.label_map_json.read_text(encoding="utf-8"))
        # Ensure background is included
        if "background" not in label_names and 0 not in label_names.values():
            label_names = {"background": 0, **label_names}
    else:
        label_names = infer_named_labels(
            train_label_values, 
            args.foreground_label_name, 
            args.label_map_json
        )

    write_dataset_json(
        dataset_dir,
        args.channel_name,
        label_names,
        len(train_ids),
        args.file_ending,
        args.dataset_name,
        args.reference,
        args.description,
    )
    print(f"✅ Prepared {dataset_dir}")


if __name__ == "__main__":
    main()
