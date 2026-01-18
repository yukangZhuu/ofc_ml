from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


# Kaggle columns use 0..94 for 95 channels, COSMOS JSON uses 1..95 for channel indices.
N_CHANNELS = 95

# This ordering matches the COSMOS codebase docs in `COSMOS-EDFA-Dataset/code/libs/edfa_feature_extraction_libs.py`.
DEFAULT_ROADM_NAMES = [
    "rdm1-co1",
    "rdm2-co1",
    "rdm3-co1",
    "rdm4-co1",
    "rdm5-co1",
    "rdm6-co1",
    "rdm1-lg1",
    "rdm2-lg1",
]


@dataclass(frozen=True)
class CosmosToKaggleConfig:
    cosmos_dataset_dir: Path
    category: str = "cosmos"
    roadm_names: tuple[str, ...] = tuple(DEFAULT_ROADM_NAMES)
    edfa_types: tuple[str, ...] = ("booster", "preamp")  # folder names in COSMOS dataset
    gains: Optional[tuple[str, ...]] = None  # e.g. ("18dB",)
    channel_types: Optional[tuple[str, ...]] = None  # e.g. ("fix","random","extraLow","extraRandom")
    max_files: Optional[int] = None
    max_records_per_file: Optional[int] = None


def _spectra_dict_to_list(d: dict[str, Any], n_channels: int = N_CHANNELS) -> list[float]:
    # JSON stores as {"1": float, ..., "95": float}; be robust to int keys.
    out: list[float] = []
    for i in range(1, n_channels + 1):
        v = d.get(str(i), d.get(i))
        if v is None:
            raise KeyError(f"Missing channel {i} in spectra dict.")
        out.append(float(v))
    return out


def _one_hot_active_channels(active_channel_indices_1_based: Iterable[int], n_channels: int = N_CHANNELS) -> list[int]:
    mask = [0] * n_channels
    for idx in active_channel_indices_1_based:
        if idx is None:
            continue
        j = int(idx) - 1
        if 0 <= j < n_channels:
            mask[j] = 1
    return mask


def _roadm_from_filename(json_path: Path) -> str:
    # Example: edfa_meas_rdm1-co1.bed_preamp_2022.02.16.21.14.46.json -> rdm1-co1
    m = re.match(r"edfa_meas_(?P<roadm>[^.]+)\.", json_path.name)
    if not m:
        raise ValueError(f"Cannot parse ROADM name from filename: {json_path.name}")
    return m.group("roadm")


def _edfa_index(roadm: str, roadm_names: tuple[str, ...]) -> int:
    if roadm not in roadm_names:
        raise ValueError(
            f"Unknown roadm '{roadm}'. Expected one of: {list(roadm_names)}. "
            "Pass `--roadm-names` / config.roadm_names to override."
        )
    return roadm_names.index(roadm)


def _kaggle_features_column_order() -> list[str]:
    cols = [
        "Category",
        "edfa_index",
        "EDFA_type",
        "target_gain",
        "target_gain_tilt",
        "EDFA_input_power_total",
        "EDFA_output_power_total",
    ]
    for i in range(N_CHANNELS):
        cols.append(f"EDFA_input_spectra_{i:02d}")
        cols.append(f"DUT_WSS_activated_channel_index_{i:02d}")
    return cols


def _kaggle_labels_column_order() -> list[str]:
    return [f"calculated_gain_spectra_{i:02d}" for i in range(N_CHANNELS)]


def _extract_one_record_booster(
    rec: dict[str, Any],
    *,
    category: str,
    edfa_type: str,
    edfa_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    info = rec["roadm_dut_edfa_info"]
    target_gain = float(info["target_gain"])
    target_gain_tilt = float(info.get("target_gain_tilt", 0.0))
    edfa_input_power_total = float(info["input_power"])
    edfa_output_power_total = float(info["output_power"])

    # Match COSMOS provided feature extraction logic:
    # - input spectra before DUT booster EDFA: `roadm_dut_wss_output_power_spectra`
    # - output spectra after DUT booster EDFA: `roadm_dut_booster_output`
    x_in = _spectra_dict_to_list(rec["roadm_dut_wss_output_power_spectra"])
    x_out = _spectra_dict_to_list(rec["roadm_dut_booster_output"])
    active = _one_hot_active_channels(rec["roadm_dut_wss_active_channel_index"])
    y_gain = [x_out[i] - x_in[i] for i in range(N_CHANNELS)]

    feat: dict[str, Any] = {
        "Category": category,
        "edfa_index": edfa_index,
        "EDFA_type": edfa_type,
        "target_gain": target_gain,
        "target_gain_tilt": target_gain_tilt,
        "EDFA_input_power_total": edfa_input_power_total,
        "EDFA_output_power_total": edfa_output_power_total,
    }
    lab: dict[str, Any] = {}

    for i in range(N_CHANNELS):
        feat[f"EDFA_input_spectra_{i:02d}"] = x_in[i]
        feat[f"DUT_WSS_activated_channel_index_{i:02d}"] = active[i]
        lab[f"calculated_gain_spectra_{i:02d}"] = y_gain[i]

    return feat, lab


def _extract_one_record_preamp(
    rec: dict[str, Any],
    *,
    category: str,
    edfa_type: str,
    edfa_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # In preamp measurement files, the DUT is in `roadm_dut_preamp_info` (and the loader config is
    # in the flatten ROADM WSS active channel list).
    info = rec["roadm_dut_preamp_info"]
    target_gain = float(info["target_gain"])
    target_gain_tilt = float(info.get("target_gain_tilt", 0.0))
    edfa_input_power_total = float(info["input_power"])
    edfa_output_power_total = float(info["output_power"])

    # Match COSMOS provided feature extraction logic:
    # - input spectra at DUT preamp input: `roadm_dut_preamp_input_power_spectra`
    # - output spectra used for gain calc: `roadm_dut_wss_input_power_spectra`
    x_in = _spectra_dict_to_list(rec["roadm_dut_preamp_input_power_spectra"])
    x_out = _spectra_dict_to_list(rec["roadm_dut_wss_input_power_spectra"])
    active = _one_hot_active_channels(rec["roadm_flatten_wss_active_channel_index"])
    y_gain = [x_out[i] - x_in[i] for i in range(N_CHANNELS)]

    feat: dict[str, Any] = {
        "Category": category,
        "edfa_index": edfa_index,
        "EDFA_type": edfa_type,
        "target_gain": target_gain,
        "target_gain_tilt": target_gain_tilt,
        "EDFA_input_power_total": edfa_input_power_total,
        "EDFA_output_power_total": edfa_output_power_total,
    }
    lab: dict[str, Any] = {}

    for i in range(N_CHANNELS):
        feat[f"EDFA_input_spectra_{i:02d}"] = x_in[i]
        feat[f"DUT_WSS_activated_channel_index_{i:02d}"] = active[i]
        lab[f"calculated_gain_spectra_{i:02d}"] = y_gain[i]

    return feat, lab


def iter_cosmos_json_files(
    dataset_dir: Path,
    *,
    edfa_types: tuple[str, ...],
    gains: Optional[tuple[str, ...]] = None,
    channel_types: Optional[tuple[str, ...]] = None,
    max_files: Optional[int] = None,
) -> Iterable[tuple[str, str, str, Path]]:
    """
    Yields (edfa_type, gain, channel_type, json_path).
    """
    n = 0
    for edfa_type in edfa_types:
        edfa_dir = dataset_dir / edfa_type
        if not edfa_dir.exists():
            continue
        for gain_dir in sorted(edfa_dir.iterdir()):
            if not gain_dir.is_dir():
                continue
            gain = gain_dir.name
            if gains is not None and gain not in gains:
                continue
            for channel_dir in sorted(gain_dir.iterdir()):
                if not channel_dir.is_dir():
                    continue
                channel_type = channel_dir.name
                if channel_types is not None and channel_type not in channel_types:
                    continue
                for json_path in sorted(channel_dir.glob("*.json")):
                    yield edfa_type, gain, channel_type, json_path
                    n += 1
                    if max_files is not None and n >= max_files:
                        return


def convert_cosmos_to_kaggle_dfs(cfg: CosmosToKaggleConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for edfa_type, _gain, _channel_type, json_path in iter_cosmos_json_files(
        cfg.cosmos_dataset_dir,
        edfa_types=cfg.edfa_types,
        gains=cfg.gains,
        channel_types=cfg.channel_types,
        max_files=cfg.max_files,
    ):
        roadm = _roadm_from_filename(json_path)
        idx = _edfa_index(roadm, cfg.roadm_names)

        with json_path.open("r") as f:
            data = json.load(f)

        measurement_data = data.get("measurement_data", [])
        if not isinstance(measurement_data, list):
            raise TypeError(f"Unexpected measurement_data type in {json_path}: {type(measurement_data)}")

        if cfg.max_records_per_file is not None:
            measurement_data = measurement_data[: cfg.max_records_per_file]

        for rec in measurement_data:
            if edfa_type == "booster":
                feat, lab = _extract_one_record_booster(
                    rec, category=cfg.category, edfa_type=edfa_type, edfa_index=idx
                )
            elif edfa_type == "preamp":
                feat, lab = _extract_one_record_preamp(
                    rec, category=cfg.category, edfa_type=edfa_type, edfa_index=idx
                )
            else:
                raise ValueError(f"Unknown edfa_type: {edfa_type}")

            feature_rows.append(feat)
            label_rows.append(lab)

    features = pd.DataFrame(feature_rows)
    labels = pd.DataFrame(label_rows)

    # Enforce Kaggle-like column order.
    features = features[_kaggle_features_column_order()]
    labels = labels[_kaggle_labels_column_order()]

    return features, labels


def write_cosmos_as_kaggle_csv(
    cfg: CosmosToKaggleConfig,
    *,
    out_features_csv: Path,
    out_labels_csv: Path,
) -> tuple[Path, Path]:
    features, labels = convert_cosmos_to_kaggle_dfs(cfg)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    out_labels_csv.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(out_features_csv, index=False)
    labels.to_csv(out_labels_csv, index=False)
    return out_features_csv, out_labels_csv

