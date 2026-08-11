"""Extract frozen DINOv2 features for paired weld images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Dinov2WithRegistersModel

from relation_tube.welding import sha256_file


PROTOCOL = "resistance_spot_welding_dinov2_v1"
REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
VIEW_ORDER = ["rgb_front", "rgb_back", "infrared"]
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]


def image_paths(dataset_root: Path, sample_ids: list[int]) -> list[Path]:
    paths = []
    for sample_id in sample_ids:
        paths.extend(
            [
                dataset_root / "Images_RGB" / f"RGB_{sample_id}F.jpg",
                dataset_root / "Images_RGB" / f"RGB_{sample_id}B.jpg",
                dataset_root / "Images_IR" / f"IR_{sample_id}.jpg",
            ]
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])
    return paths


def preprocess(paths: list[Path]) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - MEAN) / STD
        tensors.append(torch.from_numpy(array.transpose(2, 0, 1).copy()))
    return torch.stack(tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-size", default=48, type=int)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite frozen feature artifacts")
    if args.model_snapshot.name != REVISION:
        raise ValueError("DINOv2 revision changed")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    sample_ids = sorted(int(row["sample_id"]) for row in split["rows"])
    paths = image_paths(args.dataset_root, sample_ids)

    device = torch.device("cuda")
    model = Dinov2WithRegistersModel.from_pretrained(
        args.model_snapshot, local_files_only=True
    ).eval().to(device)
    model.requires_grad_(False)
    chunks = []
    for start in range(0, len(paths), args.batch_size):
        pixels = preprocess(paths[start : start + args.batch_size]).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            values = model(pixel_values=pixels).last_hidden_state[:, 0]
        chunks.append(values.float().cpu().numpy())
    features = np.concatenate(chunks).reshape(len(sample_ids), 3, -1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(sample_ids),
        view_features=features,
    )
    record = {
        "protocol": PROTOCOL,
        "status": "completed_frozen_feature_extraction",
        "model_revision": REVISION,
        "sample_count": len(sample_ids),
        "view_order": VIEW_ORDER,
        "feature_shape": list(features.shape),
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "split_sha256": sha256_file(args.split),
        "target_metrics_evaluated": False,
    }
    args.manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
