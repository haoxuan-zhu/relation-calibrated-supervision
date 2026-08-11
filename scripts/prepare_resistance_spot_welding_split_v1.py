"""Create the product-ID split used by the spot-welding benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from relation_tube.welding import stable_hash


PROTOCOL = "weld_product_split_v1"
SEED = 3407


def build_split(csv_path: Path) -> dict[str, object]:
    units: dict[int, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            units.setdefault(int(row["Sample ID"]), row)

    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for sample_id, row in units.items():
        setting = (row["Pressure (PSI)"], row["Welding Time (ms)"], row["Angle (Deg)"])
        groups[setting].append(sample_id)

    split_by_id: dict[int, str] = {}
    settings = []
    for setting in sorted(groups):
        ids = sorted(
            groups[setting],
            key=lambda sample_id: stable_hash(
                f"{SEED}|{sample_id}|{'|'.join(setting)}"
            ),
        )
        validation_count = max(1, round(0.15 * len(ids)))
        heldout_count = max(1, round(0.15 * len(ids)))
        if validation_count + heldout_count >= len(ids):
            raise ValueError(f"setting {setting} is too small for a three-way split")
        validation = ids[:validation_count]
        heldout = ids[validation_count : validation_count + heldout_count]
        train = ids[validation_count + heldout_count :]
        split_by_id.update({sample_id: "train" for sample_id in train})
        split_by_id.update({sample_id: "validation" for sample_id in validation})
        split_by_id.update({sample_id: "heldout" for sample_id in heldout})
        settings.append(
            {
                "pressure_psi": float(setting[0]),
                "welding_time_ms": float(setting[1]),
                "angle_deg": float(setting[2]),
                "total": len(ids),
                "train": len(train),
                "validation": len(validation),
                "heldout": len(heldout),
            }
        )
    counts = {
        split: sum(value == split for value in split_by_id.values())
        for split in ("train", "validation", "heldout")
    }
    return {
        "protocol": PROTOCOL,
        "seed": SEED,
        "unit": "weld_sample_id_with_three_paired_images",
        "stratification": ["Pressure (PSI)", "Welding Time (ms)", "Angle (Deg)"],
        "counts": counts,
        "setting_count": len(groups),
        "settings": settings,
        "rows": [
            {"sample_id": sample_id, "split": split_by_id[sample_id]}
            for sample_id in sorted(split_by_id)
        ],
        "heldout_targets_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite split: {args.output}")
    record = build_split(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"counts": record["counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
