import argparse
import csv
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.transform import ProjectiveTransform

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from totalspineseg.xray.labels import label_value_from_token, ordered_label_values_from_spec


SUPPORTED_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


@dataclass
class PolygonAnnotation:
    case_id: str
    order_value: float | None
    label_value: int | None
    points: list[tuple[float, float]]


# Anatomical Templates: Landmarks (0,0)-(1,1) map to the body corners.
# Points can extend outside [0, 1] to capture posterior elements.
LATERAL_VERTEBRA_TEMPLATE = [
    (0.0, 0.0), (1.0, 0.0), (1.0, 0.2), 
    (1.4, 0.5), (1.4, 0.6), (1.0, 0.8), 
    (1.0, 1.0), (0.0, 1.0), (-0.1, 0.5)
]

# AP template focuses on the body with slight transverse process hints
AP_VERTEBRA_TEMPLATE = [
    (0.0, 0.0), (0.5, -0.05), (1.0, 0.0),
    (1.1, 0.5), (1.0, 1.0), (0.5, 1.05),
    (0.0, 1.0), (-0.1, 0.5)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rasterize canonical vertebra landmark polygons into segmentation masks.",
        epilog=textwrap.dedent(
            """\
            Example:
              python scripts/xray_landmarks_to_mask.py ^
                --images-dir data/xray/images ^
                --annotations data/xray/aasce_landmarks.csv ^
                --output-dir data/xray/masks ^
                --mode binary
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder with source X-ray images.")
    parser.add_argument("--annotations", type=Path, required=True, help="CSV file with canonical vertebra polygons.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where masks will be written.")
    parser.add_argument("--case-column", default="case_id", help="CSV column used to identify each image.")
    parser.add_argument(
        "--point-columns",
        nargs=8,
        default=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"],
        help="Eight CSV columns describing the vertebra polygon.",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "sequential", "label"],
        default="binary",
        help="Binary fills all vertebrae with 1. Sequential fills 1..N after sorting. Label uses the label column.",
    )
    parser.add_argument(
        "--sort-column",
        default="vertebra_order",
        help="CSV column used to sort vertebrae for sequential masks.",
    )
    parser.add_argument(
        "--label-column",
        default="vertebra_order",
        help="CSV column used as the output value when mode=label.",
    )
    parser.add_argument(
        "--label-map-json",
        type=Path,
        default=None,
        help="Optional JSON file that maps label names such as C1 or T12 to integer ids.",
    )
    parser.add_argument(
        "--ordered-labels",
        default="",
        help=(
            "Optional superior-to-inferior label specification used by mode=label. "
            'Examples: "T1-L5" or "C1,C2,C3".'
        ),
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=False,
        help="Interpret polygon coordinates as normalized values in the range [0, 1].",
    )
    parser.add_argument(
        "--output-file-ending",
        choices=[".png", ".bmp", ".tif"],
        default=".png",
        help="Lossless file format for output masks.",
    )
    parser.add_argument(
        "--anatomical",
        action="store_true",
        default=False,
        help="Use high-fidelity anatomical templates instead of simple quadrilaterals.",
    )
    parser.add_argument(
        "--overwrite",
        "-r",
        action="store_true",
        default=False,
        help="Overwrite existing masks.",
    )
    return parser.parse_args()


def normalize_case_id(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        raise ValueError("Encountered an empty case id in the landmark CSV.")
    return Path(value).stem


def build_image_index(images_dir: Path) -> dict[str, Path]:
    image_index = {}
    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        case_id = path.stem
        if case_id in image_index:
            raise ValueError(f'Duplicate image stem "{case_id}" found under {images_dir}.')
        image_index[case_id] = path
    if not image_index:
        raise FileNotFoundError(f"No supported images were found under {images_dir}.")
    return image_index


def parse_optional_number(value: str) -> float | None:
    raw = value.strip()
    if not raw:
        return None
    return float(raw)


def parse_optional_label(value: str, label_map_json: Path | None) -> int | None:
    raw = value.strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return label_value_from_token(raw, label_map_json)


def load_annotations(
    csv_path: Path,
    case_column: str,
    point_columns: list[str],
    sort_column: str,
    label_column: str,
    label_map_json: Path | None,
) -> dict[str, list[PolygonAnnotation]]:
    grouped = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {case_column, *point_columns}
        missing = [column for column in required_columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            case_id = normalize_case_id(row[case_column])
            points = []
            valid_vertebra = True
            for x_column, y_column in zip(point_columns[::2], point_columns[1::2]):
                x_val = row.get(x_column)
                y_val = row.get(y_column)
                
                if x_val is None or y_val is None or x_val == "" or y_val == "":
                    valid_vertebra = False
                    break
                    
                try:
                    points.append((float(x_val), float(y_val)))
                except (ValueError, TypeError) as exc:
                    print(f"[!] Warning: Invalid coordinate in row {row_number} for case {case_id}: {exc}. Skipping vertebra.")
                    valid_vertebra = False
                    break

            if not valid_vertebra:
                continue

            order_value = None
            if sort_column in row:
                order_value = parse_optional_number(row[sort_column])

            label_value = None
            if label_column in row:
                label_value = parse_optional_label(row[label_column], label_map_json)

            grouped[case_id].append(
                PolygonAnnotation(
                    case_id=case_id,
                    order_value=order_value,
                    label_value=label_value,
                    points=points,
                )
            )
    if not grouped:
        raise ValueError(f"No annotations were found in {csv_path}.")
    return grouped


def clamp_point(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x, y = point
    x = min(max(int(round(x)), 0), max(width - 1, 0))
    y = min(max(int(round(y)), 0), max(height - 1, 0))
    return x, y


def scale_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    normalized: bool,
) -> list[tuple[int, int]]:
    scaled = []
    for x, y in points:
        if normalized:
            x *= width
            y *= height
        scaled.append(clamp_point((x, y), width, height))
    return scaled


def sort_annotations(annotations: list[PolygonAnnotation]) -> list[PolygonAnnotation]:
    return sorted(
        annotations,
        key=lambda item: (item.order_value is None, item.order_value if item.order_value is not None else 0),
    )


def render_mask(
    image_path: Path,
    annotations: list[PolygonAnnotation],
    normalized: bool,
    mode: str,
    ordered_label_values: list[int] | None = None,
    anatomical: bool = False,
) -> np.ndarray:
    with Image.open(image_path) as image:
        width, height = image.size
    
    # Heuristic to detect lateral vs AP based on case_id or aspect ratio
    # For now, default to Lateral style if anatomical is on.
    is_lateral = "lateral" in str(image_path).lower() or "_lat" in str(image_path).lower()
    template = LATERAL_VERTEBRA_TEMPLATE if is_lateral else AP_VERTEBRA_TEMPLATE

    sorted_annotations = sort_annotations(annotations)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for index, annotation in enumerate(sorted_annotations, start=1):
        if mode == "binary":
            fill_value = 1
        elif mode == "sequential":
            fill_value = index
        else:
            if ordered_label_values is not None:
                if len(sorted_annotations) > len(ordered_label_values):
                    raise ValueError(
                        f'Case "{annotation.case_id}" has {len(sorted_annotations)} vertebrae, '
                        f"but only {len(ordered_label_values)} ordered labels were provided."
                    )
                fill_value = ordered_label_values[index - 1]
            elif annotation.label_value is None:
                raise ValueError(
                    f'Annotation for case "{annotation.case_id}" is missing a label value required by mode=label.'
                )
            else:
                fill_value = annotation.label_value

        polygon = scale_points(annotation.points, width, height, normalized)
        
        if anatomical:
            # Warp the anatomical template to fit the 4 quadrilateral landmarks
            # Input landmarks: TL, TR, BR, BL
            # Template Body Corners: (0,0), (1,0), (1,1), (0,1)
            src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            dst = np.array(polygon, dtype=np.float32)
            
            tf = ProjectiveTransform()
            if tf.estimate(src, dst):
                # Map all template points through the transform
                warped_pts = tf(np.array(template))
                warped_polygon = [(int(round(pt[0])), int(round(pt[1]))) for pt in warped_pts]
                draw.polygon(warped_polygon, fill=fill_value)
            else:
                # Fallback to simple polygon if warping fails
                draw.polygon(polygon, fill=fill_value)
        else:
            draw.polygon(polygon, fill=fill_value)

    return np.asarray(mask, dtype=np.uint16)


def main():
    args = parse_args()

    image_index = build_image_index(args.images_dir)
    grouped_annotations = load_annotations(
        args.annotations,
        args.case_column,
        args.point_columns,
        args.sort_column,
        args.label_column,
        args.label_map_json,
    )
    ordered_label_values = None
    if args.ordered_labels:
        ordered_label_values = ordered_label_values_from_spec(args.ordered_labels, args.label_map_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_images = sorted(case_id for case_id in grouped_annotations if case_id not in image_index)
    if missing_images:
        preview = ", ".join(missing_images[:10])
        print(f"Warning: Missing source images for {len(missing_images)} annotated cases. First cases: {preview}")

    written = 0
    for case_id, annotations in grouped_annotations.items():
        output_path = args.output_dir / f"{case_id}{args.output_file_ending}"
        if output_path.exists() and not args.overwrite:
            continue
        mask = render_mask(
            image_index[case_id],
            annotations,
            args.normalized,
            args.mode,
            ordered_label_values=ordered_label_values,
            anatomical=args.anatomical,
        )
        Image.fromarray(mask).save(output_path)
        written += 1

    print(
        f"Wrote {written} masks to {args.output_dir} using mode={args.mode} "
        f"from {len(grouped_annotations)} annotated cases."
    )


if __name__ == "__main__":
    main()
