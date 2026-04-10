from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


_VERTEBRA_SEQUENCE = [
    *(f"C{i}" for i in range(1, 8)),
    *(f"T{i}" for i in range(1, 13)),
    *(f"L{i}" for i in range(1, 8)),
    "sacrum",
]
_VERTEBRA_INDEX = {name: index for index, name in enumerate(_VERTEBRA_SEQUENCE)}
_VERTEBRA_RANGE_RE = re.compile(
    r"^\s*(?P<start>(?:[CTL]\d+)|(?:sacrum))\s*-\s*(?P<end>(?:[CTL]\d+)|(?:sacrum))\s*$",
    re.IGNORECASE,
)
_INTEGER_RE = re.compile(r"^[+-]?\d+$")


@lru_cache(maxsize=1)
def totalspineseg_label_map() -> dict[str, int]:
    resource_path = Path(__file__).resolve().parents[1] / "resources" / "labels_maps" / "tss_map.json"
    with resource_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_label_map(label_map_json: str | Path | None = None) -> dict[str, int]:
    label_map = dict(totalspineseg_label_map())
    if label_map_json is None:
        return label_map

    extra_path = Path(label_map_json)
    with extra_path.open("r", encoding="utf-8") as handle:
        extra_map = json.load(handle)
    for name, value in extra_map.items():
        label_map[str(name)] = int(value)
    return label_map


def _canonical_label_alias(name: str) -> str:
    alias = name.strip().replace("_", "").replace(" ", "").upper()
    if alias in {"S", "S1", "SACRUM"}:
        return "SACRUM"
    return alias


def label_alias_map(label_map_json: str | Path | None = None) -> dict[str, int]:
    aliases = {}
    for name, value in merge_label_map(label_map_json).items():
        aliases[_canonical_label_alias(name)] = int(value)
    return aliases


def label_name_map(label_map_json: str | Path | None = None) -> dict[int, str]:
    return {int(value): name for name, value in merge_label_map(label_map_json).items()}


def normalize_vertebra_name(name: str) -> str:
    raw = name.strip()
    if not raw:
        raise ValueError("Vertebra name cannot be empty.")

    if raw.lower() == "sacrum":
        return "sacrum"

    match = re.fullmatch(r"([CTLctl])\s*0*([1-9]\d*)", raw)
    if not match:
        raise ValueError(f'Unsupported vertebra name "{name}".')
    region = match.group(1).upper()
    index = int(match.group(2))
    normalized = f"{region}{index}"
    if normalized not in _VERTEBRA_INDEX:
        raise ValueError(f'Unsupported vertebra name "{name}".')
    return normalized


def vertebra_names_from_spec(spec: str) -> list[str]:
    value = spec.strip()
    if not value:
        raise ValueError("ordered label specification cannot be empty.")

    range_match = _VERTEBRA_RANGE_RE.fullmatch(value)
    if range_match:
        start = normalize_vertebra_name(range_match.group("start"))
        end = normalize_vertebra_name(range_match.group("end"))
        start_index = _VERTEBRA_INDEX[start]
        end_index = _VERTEBRA_INDEX[end]
        if start_index > end_index:
            raise ValueError(f'Ordered label range "{spec}" must be superior-to-inferior.')
        return _VERTEBRA_SEQUENCE[start_index : end_index + 1]

    return [normalize_vertebra_name(token) for token in value.split(",") if token.strip()]


def ordered_label_values_from_spec(spec: str, label_map_json: str | Path | None = None) -> list[int]:
    label_map = merge_label_map(label_map_json)
    values = []
    for name in vertebra_names_from_spec(spec):
        if name not in label_map:
            raise ValueError(f'No label value is defined for vertebra "{name}".')
        values.append(int(label_map[name]))
    return values


def label_value_from_token(token: str, label_map_json: str | Path | None = None) -> int:
    raw = token.strip()
    if not raw:
        raise ValueError("Label token cannot be empty.")

    if _INTEGER_RE.fullmatch(raw):
        return int(raw)

    aliases = label_alias_map(label_map_json)
    alias = _canonical_label_alias(raw)
    if alias not in aliases:
        raise ValueError(f'Unknown label token "{token}".')
    return aliases[alias]


def infer_named_labels(
    label_values: set[int],
    foreground_name: str = "vertebrae",
    label_map_json: str | Path | None = None,
) -> dict[str, int]:
    positive_values = sorted(int(value) for value in label_values if int(value) > 0)
    if not positive_values:
        raise ValueError("No positive label values were found in the training masks.")

    if positive_values == [1]:
        return {"background": 0, foreground_name: 1}

    known_names = label_name_map(label_map_json)
    labels = {"background": 0}
    for value in positive_values:
        labels[known_names.get(value, f"label_{value}")] = value
    return labels


def align_ordered_label_values(
    label_values: list[int],
    num_components: int,
    anchor: str = "strict",
) -> list[int]:
    if num_components <= 0:
        return []
    if num_components > len(label_values):
        raise ValueError(
            f"Requested {num_components} ordered labels but only {len(label_values)} were provided."
        )

    if anchor == "strict":
        if num_components != len(label_values):
            raise ValueError(
                f"Strict ordered labeling requires {len(label_values)} components but found {num_components}."
            )
        return list(label_values)
    if anchor == "superior":
        return list(label_values[:num_components])
    if anchor == "inferior":
        return list(label_values[-num_components:])
    raise ValueError(f'Unsupported ordered label anchor "{anchor}".')
