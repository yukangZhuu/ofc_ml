#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script without installing the package (repo layout: src/ofc_ml).
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ofc_ml.cosmos_to_kaggle import CosmosToKaggleConfig, write_cosmos_as_kaggle_csv  # noqa: E402


def _parse_csv_list(s: str | None) -> tuple[str, ...] | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return tuple(x.strip() for x in s.split(",") if x.strip())


def main() -> None:
    p = argparse.ArgumentParser(description="Convert COSMOS-EDFA JSON dataset into Kaggle-style CSV features/labels.")
    p.add_argument(
        "--cosmos-dataset-dir",
        type=Path,
        default=Path("/home/shaowen/ofc_ml/COSMOS-EDFA-Dataset/dataset"),
        help="Path to COSMOS-EDFA-Dataset/dataset",
    )
    p.add_argument("--out-dir", type=Path, default=Path("/home/shaowen/ofc_ml/data/cosmos-as-kaggle"))
    p.add_argument("--category", type=str, default="unseen", help="Value to fill in Kaggle 'Category' column.")
    p.add_argument(
        "--edfa-types",
        type=str,
        default="booster,preamp",
        help="Comma-separated subset of {booster,preamp}.",
    )
    p.add_argument("--gains", type=str, default=None, help="Comma-separated gains to include, e.g. '18dB'.")
    p.add_argument(
        "--channel-types",
        type=str,
        default=None,
        help="Comma-separated channel types to include, e.g. 'fix,random,extraLow,extraRandom'.",
    )
    p.add_argument("--max-files", type=int, default=None, help="Limit number of JSON files (debug).")
    p.add_argument("--max-records-per-file", type=int, default=None, help="Limit records per JSON file (debug).")

    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_features = out_dir / "train_features.csv"
    out_labels = out_dir / "train_labels.csv"

    cfg = CosmosToKaggleConfig(
        cosmos_dataset_dir=args.cosmos_dataset_dir,
        category=args.category,
        edfa_types=_parse_csv_list(args.edfa_types) or ("booster", "preamp"),
        gains=_parse_csv_list(args.gains),
        channel_types=_parse_csv_list(args.channel_types),
        max_files=args.max_files,
        max_records_per_file=args.max_records_per_file,
    )

    write_cosmos_as_kaggle_csv(cfg, out_features_csv=out_features, out_labels_csv=out_labels)
    print(f"Wrote: {out_features}")
    print(f"Wrote: {out_labels}")


if __name__ == "__main__":
    main()

