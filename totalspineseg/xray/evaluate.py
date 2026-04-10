import argparse
import csv
import statistics
import textwrap
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from .common import SUPPORTED_LABEL_SUFFIXES, collect_case_files, load_label_image, write_json
from .labels import label_name_map


def safe_metric(numerator: int, denominator: int, pred_positive: int, label_positive: int) -> float:
    if denominator > 0:
        return float(numerator) / float(denominator)
    return 1.0 if pred_positive == 0 and label_positive == 0 else 0.0


def metric_dict(prediction_positive: np.ndarray, reference_positive: np.ndarray) -> dict[str, float | int]:
    tp = int(np.logical_and(prediction_positive, reference_positive).sum())
    fp = int(np.logical_and(prediction_positive, ~reference_positive).sum())
    fn = int(np.logical_and(~prediction_positive, reference_positive).sum())

    pred_area = int(prediction_positive.sum())
    ref_area = int(reference_positive.sum())

    dice = safe_metric(2 * tp, 2 * tp + fp + fn, pred_area, ref_area)
    iou = safe_metric(tp, tp + fp + fn, pred_area, ref_area)
    precision = safe_metric(tp, tp + fp, pred_area, ref_area)
    recall = safe_metric(tp, tp + fn, pred_area, ref_area)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_area": pred_area,
        "ref_area": ref_area,
    }


def compute_binary_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    metrics = metric_dict(prediction > 0, reference > 0)
    _, pred_components = ndi.label(prediction > 0)
    _, ref_components = ndi.label(reference > 0)
    metrics["pred_components"] = int(pred_components)
    metrics["ref_components"] = int(ref_components)
    return metrics


def compute_multiclass_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    known_label_names: dict[int, str],
) -> tuple[dict[str, float | int], list[dict[str, float | int | str | None]]]:
    label_values = sorted(
        int(value)
        for value in (set(np.unique(prediction).tolist()) | set(np.unique(reference).tolist()))
        if int(value) > 0
    )
    per_label_rows = []
    for label_value in label_values:
        metrics = metric_dict(prediction == label_value, reference == label_value)
        per_label_rows.append(
            {
                "label_value": label_value,
                "label_name": known_label_names.get(label_value),
                **metrics,
            }
        )

    if per_label_rows:
        summary = {
            "label_dice": statistics.fmean(float(row["dice"]) for row in per_label_rows),
            "label_iou": statistics.fmean(float(row["iou"]) for row in per_label_rows),
            "label_precision": statistics.fmean(float(row["precision"]) for row in per_label_rows),
            "label_recall": statistics.fmean(float(row["recall"]) for row in per_label_rows),
            "num_labels_evaluated": len(per_label_rows),
            "label_set_match": int(
                {int(value) for value in np.unique(prediction) if int(value) > 0}
                == {int(value) for value in np.unique(reference) if int(value) > 0}
            ),
        }
    else:
        summary = {
            "label_dice": 0.0,
            "label_iou": 0.0,
            "label_precision": 0.0,
            "label_recall": 0.0,
            "num_labels_evaluated": 0,
            "label_set_match": 1,
        }
    return summary, per_label_rows


def summarize_metrics(rows: list[dict[str, float | int]], metric_names: list[str]) -> dict[str, object]:
    summary = {"num_cases": len(rows)}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        summary[metric_name] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return summary


def evaluate_folder(
    prediction_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    prediction_suffix: str = "",
    reference_suffix: str = "",
    fail_on_missing: bool = True,
    evaluation_mode: str = "auto",
    label_map_json: str | Path | None = None,
):
    predictions = collect_case_files(prediction_dir, SUPPORTED_LABEL_SUFFIXES, prediction_suffix)
    references = collect_case_files(reference_dir, SUPPORTED_LABEL_SUFFIXES, reference_suffix)
    known_label_names = label_name_map(label_map_json)

    matched_case_ids = sorted(set(predictions) & set(references))
    missing_predictions = sorted(set(references) - set(predictions))
    missing_references = sorted(set(predictions) - set(references))

    if fail_on_missing and (missing_predictions or missing_references):
        problems = []
        if missing_predictions:
            problems.append(f"missing predictions: {', '.join(missing_predictions[:10])}")
        if missing_references:
            problems.append(f"missing references: {', '.join(missing_references[:10])}")
        raise ValueError("; ".join(problems))

    per_case_rows = []
    per_label_rows = []
    all_label_values = set()

    for case_id in matched_case_ids:
        prediction = load_label_image(predictions[case_id])
        reference = load_label_image(references[case_id])
        if prediction.shape != reference.shape:
            raise ValueError(
                f'Shape mismatch for case "{case_id}": prediction={prediction.shape}, reference={reference.shape}'
            )

        binary_metrics = compute_binary_metrics(prediction, reference)
        row = {"case_id": case_id, **binary_metrics}

        positive_labels = {
            int(value)
            for value in (set(np.unique(prediction).tolist()) | set(np.unique(reference).tolist()))
            if int(value) > 0
        }
        if evaluation_mode == "multiclass" or (evaluation_mode == "auto" and len(positive_labels) > 1):
            label_summary, case_label_rows = compute_multiclass_metrics(prediction, reference, known_label_names)
            row.update(label_summary)
            per_label_rows.extend({"case_id": case_id, **label_row} for label_row in case_label_rows)
            all_label_values.update(positive_labels)

        per_case_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_csv = output_dir / "per_case_metrics.csv"
    per_case_fields = [
        "case_id",
        "dice",
        "iou",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "pred_area",
        "ref_area",
        "pred_components",
        "ref_components",
    ]
    if per_label_rows or evaluation_mode == "multiclass":
        per_case_fields.extend(
            [
                "label_dice",
                "label_iou",
                "label_precision",
                "label_recall",
                "num_labels_evaluated",
                "label_set_match",
            ]
        )
    with per_case_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_case_fields)
        writer.writeheader()
        writer.writerows(per_case_rows)

    if per_label_rows:
        per_label_csv = output_dir / "per_label_metrics.csv"
        with per_label_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "case_id",
                    "label_value",
                    "label_name",
                    "dice",
                    "iou",
                    "precision",
                    "recall",
                    "tp",
                    "fp",
                    "fn",
                    "pred_area",
                    "ref_area",
                ],
            )
            writer.writeheader()
            writer.writerows(per_label_rows)

    summary = summarize_metrics(per_case_rows, ["dice", "iou", "precision", "recall"])
    if per_label_rows or evaluation_mode == "multiclass":
        label_summary = summarize_metrics(
            [row for row in per_case_rows if "label_dice" in row],
            ["label_dice", "label_iou", "label_precision", "label_recall", "label_set_match"],
        )
        summary.update({key: value for key, value in label_summary.items() if key != "num_cases"})
        summary["labels_evaluated"] = sorted(all_label_values)

    summary["evaluation_mode"] = evaluation_mode
    summary["label_map_json"] = None if label_map_json is None else str(label_map_json)
    summary["missing_predictions"] = missing_predictions
    summary["missing_references"] = missing_references
    write_json(summary, output_dir / "metrics_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate X-ray segmentation predictions against reference masks.",
        epilog=textwrap.dedent(
            """\
            Examples:
              totalspineseg_xray_evaluate preds labels report
              totalspineseg_xray_evaluate preds labels report --evaluation-mode multiclass
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("prediction_dir", type=Path, help="Folder with predicted masks.")
    parser.add_argument("reference_dir", type=Path, help="Folder with reference masks.")
    parser.add_argument("output_dir", type=Path, help="Folder where reports will be written.")
    parser.add_argument("--prediction-suffix", default="", help="Suffix stripped from prediction stems.")
    parser.add_argument("--reference-suffix", default="", help="Suffix stripped from reference stems.")
    parser.add_argument(
        "--evaluation-mode",
        choices=["auto", "binary", "multiclass"],
        default="auto",
        help="Whether to evaluate foreground masks only or also compute per-label metrics.",
    )
    parser.add_argument(
        "--label-map-json",
        type=Path,
        default=None,
        help="Optional JSON file that maps anatomical label names to integer ids.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        default=False,
        help="Do not fail if cases are missing on either side.",
    )
    args = parser.parse_args()

    evaluate_folder(
        prediction_dir=args.prediction_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        prediction_suffix=args.prediction_suffix,
        reference_suffix=args.reference_suffix,
        fail_on_missing=not args.allow_missing,
        evaluation_mode=args.evaluation_mode,
        label_map_json=args.label_map_json,
    )


if __name__ == "__main__":
    main()
