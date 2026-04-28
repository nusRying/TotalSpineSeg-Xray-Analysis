"""
TotalSpineSeg X-Ray: Production Post-processing Engine
Version: 2026.04.25 (Milestone 4)

This script implements clinical post-processing for spine X-ray masks, including:
1. Nearest-Neighbor Rescaling for Resolution Restoration
2. Connected Component Extraction for Vertebrae
3. Automated Superior-to-Inferior Label Assignment (Anatomical Numbering)
4. Overlay Generation for Clinical Review
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from .common import (
    SUPPORTED_IMAGE_SUFFIXES,
    SUPPORTED_LABEL_SUFFIXES,
    collect_case_files,
    load_label_image,
    save_label_image,
    save_overlay_preview,
    write_json,
)
from PIL import Image
from .labels import (
    align_ordered_label_values,
    label_name_map,
    ordered_label_values_from_spec,
)


def component_bbox(component_mask: np.ndarray) -> list[int]:
    y_coords, x_coords = np.where(component_mask)
    return [
        int(y_coords.min()),
        int(x_coords.min()),
        int(y_coords.max()),
        int(x_coords.max()),
    ]


def apply_watershed_separation(mask: np.ndarray, min_distance: int = 40) -> np.ndarray:
    """
    Separates merged vertebrae components using distance transform and watershed.
    """
    if not mask.any():
        return mask
    
    # 1. Distance transform: how far is each pixel from the background?
    distance = ndi.distance_transform_edt(mask)
    
    # 2. Find local maxima (peaks): these are our vertebra centers
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask)
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)
    
    # 3. Apply watershed to separate components based on the peaks
    labels = watershed(-distance, markers, mask=mask)
    return labels


def extract_component_entries(
    binary_mask: np.ndarray,
    min_area: int,
    fill_holes: bool,
    max_components: int | None = None,
    label_value: int | None = None,
    label_name: str | None = None,
) -> tuple[list[dict[str, object]], int]:
    if fill_holes:
        binary_mask = ndi.binary_fill_holes(binary_mask)

    # Apply Watershed separation to solve "Thoracic Blob" merging issues
    separated_labels = apply_watershed_separation(binary_mask)
    num_components = int(separated_labels.max())
    
    entries = []
    for component_id in range(1, num_components + 1):
        component_mask = separated_labels == component_id
        area = int(component_mask.sum())
        if area < min_area:
            continue
        center_y, center_x = ndi.center_of_mass(component_mask)
        summary = {
            "component_id": component_id,
            "area": area,
            "centroid_y": float(center_y),
            "centroid_x": float(center_x),
            "bbox": component_bbox(component_mask),
        }
        if label_value is not None:
            summary["label_value"] = int(label_value)
        if label_name is not None:
            summary["label_name"] = label_name
        entries.append({"mask": component_mask, "summary": summary})

    if max_components is not None and len(entries) > max_components:
        entries = sorted(entries, key=lambda item: item["summary"]["area"], reverse=True)[:max_components]

    return entries, int(num_components)


def finalize_component_entries(
    entries: list[dict[str, object]],
    output_shape: tuple[int, ...],
    known_label_names: dict[int, str],
    assigned_label_values: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    ordered = np.zeros(output_shape, dtype=np.uint16)
    labeled = np.zeros(output_shape, dtype=np.uint16)

    sorted_entries = sorted(
        entries,
        key=lambda item: (
            item["summary"]["centroid_y"],
            item["summary"]["centroid_x"],
        ),
    )

    summaries: list[dict[str, object]] = []
    for ordered_label, entry in enumerate(sorted_entries, start=1):
        component_mask = entry["mask"]
        component_summary = dict(entry["summary"])
        component_summary["ordered_label"] = ordered_label
        ordered[component_mask] = ordered_label

        label_value = component_summary.get("label_value")
        if assigned_label_values is not None:
            label_value = int(assigned_label_values[ordered_label - 1])
            component_summary["label_value"] = label_value
        if label_value is not None:
            label_value = int(label_value)
            labeled[component_mask] = label_value
            component_summary["label_value"] = label_value
            label_name = known_label_names.get(label_value)
            if label_name is not None:
                component_summary["label_name"] = label_name

        summaries.append(component_summary)

    cleaned_binary = (ordered > 0).astype(np.uint8)
    return cleaned_binary, ordered, labeled, summaries


def postprocess_binary_prediction(
    prediction: np.ndarray,
    min_area: int,
    max_components: int | None,
    fill_holes: bool,
    known_label_names: dict[int, str],
    ordered_label_values: list[int] | None,
    ordered_label_anchor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    entries, num_components_raw = extract_component_entries(
        prediction > 0,
        min_area=min_area,
        fill_holes=fill_holes,
        max_components=max_components,
    )

    assigned_label_values = None
    assignment_mode = "none"
    if ordered_label_values is not None:
        assigned_label_values = align_ordered_label_values(
            ordered_label_values,
            num_components=len(entries),
            anchor=ordered_label_anchor,
        )
        assignment_mode = ordered_label_anchor

    cleaned_binary, ordered, labeled, component_summaries = finalize_component_entries(
        entries,
        output_shape=prediction.shape,
        known_label_names=known_label_names,
        assigned_label_values=assigned_label_values,
    )
    summary = {
        "prediction_mode": "binary",
        "num_components_raw": int(num_components_raw),
        "num_components_kept": int(len(component_summaries)),
        "num_components_removed": int(num_components_raw - len(component_summaries)),
        "min_area": int(min_area),
        "max_components": None if max_components is None else int(max_components),
        "fill_holes": bool(fill_holes),
        "ordered_label_assignment": assignment_mode,
        "components": component_summaries,
    }
    return cleaned_binary, ordered, labeled, summary


def postprocess_multiclass_prediction(
    prediction: np.ndarray,
    min_area: int,
    fill_holes: bool,
    known_label_names: dict[int, str],
    max_components_per_label: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    positive_values = sorted(int(value) for value in np.unique(prediction) if int(value) > 0)
    all_entries: list[dict[str, object]] = []
    label_summaries = []
    raw_components_total = 0

    for label_value in positive_values:
        label_entries, raw_components = extract_component_entries(
            prediction == label_value,
            min_area=min_area,
            fill_holes=fill_holes,
            max_components=max_components_per_label,
            label_value=label_value,
            label_name=known_label_names.get(label_value),
        )
        raw_components_total += raw_components
        label_summaries.append(
            {
                "label_value": label_value,
                "label_name": known_label_names.get(label_value),
                "num_components_raw": int(raw_components),
                "num_components_kept": int(len(label_entries)),
            }
        )
        all_entries.extend(label_entries)

    cleaned_binary, ordered, labeled, component_summaries = finalize_component_entries(
        all_entries,
        output_shape=prediction.shape,
        known_label_names=known_label_names,
    )
    summary = {
        "prediction_mode": "multiclass",
        "num_components_raw": int(raw_components_total),
        "num_components_kept": int(len(component_summaries)),
        "num_components_removed": int(raw_components_total - len(component_summaries)),
        "min_area": int(min_area),
        "max_components_per_label": None if max_components_per_label is None else int(max_components_per_label),
        "fill_holes": bool(fill_holes),
        "ordered_label_assignment": "raw_multiclass",
        "labels_present": positive_values,
        "labels": label_summaries,
        "components": component_summaries,
    }
    return cleaned_binary, ordered, labeled, summary


def postprocess_prediction(
    prediction: np.ndarray,
    min_area: int = 64,
    max_components: int | None = None,
    fill_holes: bool = True,
    prediction_mode: str = "auto",
    ordered_label_values: list[int] | None = None,
    ordered_label_anchor: str = "strict",
    label_map_json: str | Path | None = None,
    max_components_per_label: int | None = 1,
):
    known_label_names = label_name_map(label_map_json)
    positive_values = sorted(int(value) for value in np.unique(prediction) if int(value) > 0)
    if prediction_mode == "auto":
        effective_mode = "multiclass" if len(positive_values) > 1 else "binary"
    else:
        effective_mode = prediction_mode

    if effective_mode == "multiclass":
        return postprocess_multiclass_prediction(
            prediction=prediction,
            min_area=min_area,
            fill_holes=fill_holes,
            known_label_names=known_label_names,
            max_components_per_label=max_components_per_label,
        )
    if effective_mode == "binary":
        return postprocess_binary_prediction(
            prediction=prediction,
            min_area=min_area,
            max_components=max_components,
            fill_holes=fill_holes,
            known_label_names=known_label_names,
            ordered_label_values=ordered_label_values,
            ordered_label_anchor=ordered_label_anchor,
        )
    raise ValueError(f'Unsupported prediction mode "{prediction_mode}".')


def postprocess_folder(
    input_dir: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    staged_shapes: dict[str, tuple[int, int]] | None = None,
    prediction_suffix: str = "",
    image_suffix: str = "",
    file_ending: str = ".png",
    min_area: int = 64,
    max_components: int | None = None,
    fill_holes: bool = True,
    overwrite: bool = False,
    prediction_mode: str = "auto",
    ordered_label_values: list[int] | None = None,
    ordered_label_anchor: str = "strict",
    label_map_json: str | Path | None = None,
    max_components_per_label: int | None = 1,
):
    predictions = collect_case_files(input_dir, SUPPORTED_LABEL_SUFFIXES, prediction_suffix)
    images = None
    if image_dir is not None:
        images = collect_case_files(image_dir, SUPPORTED_IMAGE_SUFFIXES, image_suffix)

    binary_dir = output_dir / "binary"
    ordered_dir = output_dir / "ordered"
    labeled_dir = output_dir / "labeled"
    preview_dir = output_dir / "preview"

    case_summaries = []
    expects_labeled_output = ordered_label_values is not None or prediction_mode != "binary"
    for case_id, prediction_path in sorted(predictions.items()):
        if images is not None and case_id not in images:
            raise FileNotFoundError(f'Missing image for prediction case "{case_id}" in {image_dir}')

        binary_path = binary_dir / f"{case_id}{file_ending}"
        ordered_path = ordered_dir / f"{case_id}{file_ending}"
        labeled_path = labeled_dir / f"{case_id}{file_ending}"
        preview_path = preview_dir / f"{case_id}{file_ending}"

        if (
            binary_path.exists()
            and ordered_path.exists()
            and (not expects_labeled_output or labeled_path.exists())
            and not overwrite
        ):
            continue

        prediction = load_label_image(prediction_path)

        # Rescale if needed
        if staged_shapes and case_id in staged_shapes:
            target_width, target_height = staged_shapes[case_id]
            if (target_height, target_width) != prediction.shape:
                pred_image = Image.fromarray(prediction)
                # Resampling to nearest neighbor is critical for label maps
                pred_image = pred_image.resize((target_width, target_height), resample=Image.Resampling.NEAREST)
                prediction = np.asarray(pred_image)

        cleaned_binary, ordered, labeled, summary = postprocess_prediction(
            prediction=prediction,
            min_area=min_area,
            max_components=max_components,
            fill_holes=fill_holes,
            prediction_mode=prediction_mode,
            ordered_label_values=ordered_label_values,
            ordered_label_anchor=ordered_label_anchor,
            label_map_json=label_map_json,
            max_components_per_label=max_components_per_label,
        )
        summary["case_id"] = case_id

        save_label_image(cleaned_binary, binary_path)
        save_label_image(ordered, ordered_path)
        if labeled.any():
            save_label_image(labeled, labeled_path)
        elif labeled_path.exists() and overwrite:
            labeled_path.unlink()

        if images is not None:
            overlay_mask = labeled if labeled.any() else ordered
            annotations = []
            for comp in summary["components"]:
                text = comp.get("label_name")
                if not text:
                     # Fallback to ordered label if no anatomical name is assigned
                     text = str(comp.get("ordered_label", ""))
                
                if text:
                    annotations.append(
                        {
                            "text": text,
                            "centroid": (comp["centroid_y"], comp["centroid_x"]),
                        }
                    )
            save_overlay_preview(images[case_id], overlay_mask, preview_path, annotations=annotations)

        case_summaries.append(summary)

    output_summary = {
        "num_cases": len(case_summaries),
        "prediction_mode": prediction_mode,
        "min_area": int(min_area),
        "max_components": None if max_components is None else int(max_components),
        "max_components_per_label": None if max_components_per_label is None else int(max_components_per_label),
        "fill_holes": bool(fill_holes),
        "ordered_label_anchor": ordered_label_anchor,
        "ordered_label_values": ordered_label_values,
        "label_map_json": None if label_map_json is None else str(label_map_json),
        "cases": case_summaries,
    }
    write_json(output_summary, output_dir / "postprocess_summary.json")
    return output_summary


def main():
    parser = argparse.ArgumentParser(
        description="Clean X-ray predictions, order vertebrae, and optionally assign anatomical labels.",
        epilog=textwrap.dedent(
            """\
            Examples:
              totalspineseg_xray_postprocess raw_preds output --image-dir images --min-area 128
              totalspineseg_xray_postprocess raw_preds output --prediction-mode binary --ordered-labels T1-L5
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing raw predicted label masks.")
    parser.add_argument("output_dir", type=Path, help="Folder where processed outputs will be written.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional folder with source X-ray images.")
    parser.add_argument(
        "--staged-shapes-json",
        type=Path,
        default=None,
        help="Optional JSON file mapping case ids to original (width, height) shapes for adaptive rescaling.",
    )
    parser.add_argument("--prediction-suffix", default="", help="Suffix stripped from prediction stems.")
    parser.add_argument("--image-suffix", default="", help="Suffix stripped from image stems.")
    parser.add_argument("--file-ending", choices=[".png", ".bmp", ".tif"], default=".png")
    parser.add_argument("--min-area", type=int, default=64, help="Minimum component area in pixels.")
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
        help="How to interpret the raw prediction values before postprocessing.",
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
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        default=False,
        help="Disable binary hole filling before component extraction.",
    )
    parser.add_argument("--overwrite", "-r", action="store_true", default=False)
    args = parser.parse_args()

    ordered_label_values = None
    if args.ordered_labels:
        ordered_label_values = ordered_label_values_from_spec(args.ordered_labels, args.label_map_json)

    staged_shapes = None
    if args.staged_shapes_json:
        with open(args.staged_shapes_json, "r") as f:
            raw_shapes = json.load(f)
            # Handle manifest format if provided
            if "staged_shapes" in raw_shapes:
                raw_shapes = raw_shapes["staged_shapes"]
            staged_shapes = {k: tuple(v) for k, v in raw_shapes.items()}

    postprocess_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        staged_shapes=staged_shapes,
        prediction_suffix=args.prediction_suffix,
        image_suffix=args.image_suffix,
        file_ending=args.file_ending,
        min_area=args.min_area,
        max_components=args.max_components,
        fill_holes=not args.no_fill_holes,
        overwrite=args.overwrite,
        prediction_mode=args.prediction_mode,
        ordered_label_values=ordered_label_values,
        ordered_label_anchor=args.ordered_label_anchor,
        label_map_json=args.label_map_json,
        max_components_per_label=args.max_components_per_label,
    )


if __name__ == "__main__":
    main()
