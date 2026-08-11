"""Extract frozen pair, object and mask-motion features for formal MOVi evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


PROTOCOL = "movi_collision_multiview_dinov2_v1"
BACKBONE_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected MOVi multiview feature protocol")
    data = config["data"]
    if data.get("heldout_target_evaluated") is not False:
        raise ValueError("feature export must not evaluate held-out targets")
    backbone = config["backbone"]
    if backbone["revision"] != BACKBONE_REVISION:
        raise ValueError("backbone revision changed")
    if int(backbone["hidden_size"]) != 768 or int(backbone["input_size"]) != 224:
        raise ValueError("backbone dimensions changed")
    if [int(value) for value in backbone["pair_neutral_rgb"]] != [127, 127, 127]:
        raise ValueError("registered pair neutral color changed")
    if [int(value) for value in backbone["object_neutral_rgb"]] != [128, 128, 128]:
        raise ValueError("registered object neutral color changed")
    if int(backbone["pair_crop_padding_pixels"]) != 8:
        raise ValueError("registered pair crop padding changed")
    if int(backbone["object_crop_padding_pixels"]) != 4:
        raise ValueError("registered object crop padding changed")


def masked_images(
    rgb: np.ndarray,
    masks: np.ndarray,
    *,
    selected_values: tuple[int, ...],
    neutral_rgb: tuple[int, int, int],
    padding: int,
    output_size: int,
) -> list[Image.Image]:
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or masks.shape != rgb.shape[:3]:
        raise ValueError("invalid MOVi clip arrays")
    selected = np.isin(masks, np.asarray(selected_values, dtype=masks.dtype))
    union = np.any(selected, axis=0)
    rows, columns = np.where(union)
    if rows.size == 0:
        raise ValueError("selected object mask is empty")
    top = max(0, int(rows.min()) - padding)
    bottom = min(rgb.shape[1], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(rgb.shape[2], int(columns.max()) + padding + 1)
    height, width = bottom - top, right - left
    side = max(height, width)
    neutral_value = np.asarray(neutral_rgb, dtype=np.uint8)
    images: list[Image.Image] = []
    for frame, keep in zip(rgb, selected, strict=True):
        neutral = np.empty_like(frame)
        neutral[...] = neutral_value
        neutral[keep] = frame[keep]
        crop = neutral[top:bottom, left:right]
        square = np.empty((side, side, 3), dtype=np.uint8)
        square[...] = neutral_value
        y = (side - height) // 2
        x = (side - width) // 2
        square[y : y + height, x : x + width] = crop
        images.append(
            Image.fromarray(square, mode="RGB").resize(
                (output_size, output_size), Image.Resampling.BICUBIC
            )
        )
    return images


def mask_motion_features(rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or masks.shape != rgb.shape[:3]:
        raise ValueError("invalid MOVi clip arrays")
    height, width = masks.shape[1:]
    rows: list[list[float]] = []
    for frame_rgb, frame_mask in zip(rgb, masks, strict=True):
        for value in (1, 2):
            y, x = np.where(frame_mask == value)
            if y.size == 0:
                raise ValueError("object missing from registered frame")
            pixels = frame_rgb[y, x].astype(np.float64) / 255.0
            rows.append(
                [
                    float((x.mean() + 0.5) / width),
                    float((y.mean() + 0.5) / height),
                    float(y.size / (height * width)),
                    float((x.max() - x.min() + 1) / width),
                    float((y.max() - y.min() + 1) / height),
                    *pixels.mean(axis=0).tolist(),
                ]
            )
    return np.asarray(rows, dtype=np.float32).reshape(3, 2, 8)


def extract(config: dict[str, Any], clip_path: Path, output: Path) -> dict[str, Any]:
    validate_config(config)
    clip_hash = sha256_file(clip_path)
    if clip_hash != str(config["data"]["clip_npz_sha256"]):
        raise ValueError("formal compact archive hash does not match")
    with np.load(clip_path, allow_pickle=False) as clips:
        names = np.asarray(clips["video_name"])
        rgb = np.asarray(clips["rgb"], dtype=np.uint8)
        masks = np.asarray(clips["pair_mask"], dtype=np.uint8)
    if rgb.shape[:2] != (len(names), 3) or masks.shape != rgb.shape[:-1]:
        raise ValueError("formal clip arrays have unexpected shapes")

    try:
        import torch
        from transformers import AutoImageProcessor, Dinov2WithRegistersModel
    except ImportError as error:
        raise RuntimeError("server feature dependencies are unavailable") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal DINOv2 export")
    backbone = config["backbone"]
    processor = AutoImageProcessor.from_pretrained(
        backbone["model_id"],
        revision=backbone["revision"],
        local_files_only=True,
        use_fast=False,
    )
    model = Dinov2WithRegistersModel.from_pretrained(
        backbone["model_id"],
        revision=backbone["revision"],
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval().cuda()
    hidden = int(backbone["hidden_size"])
    pair_neutral = tuple(int(value) for value in backbone["pair_neutral_rgb"])
    object_neutral = tuple(int(value) for value in backbone["object_neutral_rgb"])
    pair_padding = int(backbone["pair_crop_padding_pixels"])
    object_padding = int(backbone["object_crop_padding_pixels"])
    image_size = int(backbone["input_size"])
    event_batch = int(backbone["batch_size_events"])
    encoded = np.empty((len(names), 9, hidden), dtype=np.float32)
    geometry = np.empty((len(names), 3, 2, 8), dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0, len(names), event_batch):
            end = min(len(names), begin + event_batch)
            images: list[Image.Image] = []
            for index in range(begin, end):
                images.extend(
                    masked_images(
                        rgb[index],
                        masks[index],
                        selected_values=(1, 2),
                        neutral_rgb=pair_neutral,
                        padding=pair_padding,
                        output_size=image_size,
                    )
                )
                for value in (1, 2):
                    images.extend(
                        masked_images(
                            rgb[index],
                            masks[index],
                            selected_values=(value,),
                            neutral_rgb=object_neutral,
                            padding=object_padding,
                            output_size=image_size,
                        )
                    )
                geometry[index] = mask_motion_features(rgb[index], masks[index])
            pixels = processor(
                images=images,
                do_resize=False,
                do_center_crop=False,
                return_tensors="pt",
            )["pixel_values"].cuda(non_blocking=True)
            if tuple(pixels.shape[-2:]) != (image_size, image_size):
                raise RuntimeError("processor changed the frozen input shape")
            values = model(pixel_values=pixels).last_hidden_state[:, 0, :]
            encoded[begin:end] = values.float().cpu().numpy().reshape(end - begin, 9, hidden)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            video_name=names,
            pair_cls=encoded[:, :3],
            object_cls=encoded[:, 3:].reshape(len(names), 2, 3, hidden),
            mask_motion=geometry,
        )
    temporary.replace(output)
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_formal_frozen_multiview_feature_export",
        "benchmark_metrics": False,
        "heldout_target_evaluated": False,
        "events": len(names),
        "clip_npz_sha256": clip_hash,
        "feature_npz": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "pair_cls_shape": list(encoded[:, :3].shape),
            "object_cls_shape": [len(names), 2, 3, hidden],
            "mask_motion_shape": list(geometry.shape),
        },
        "backbone": {
            "model_id": backbone["model_id"],
            "revision": backbone["revision"],
            "updated": False,
        },
        "preprocess": {
            "shared_temporal_crop": True,
            "pair_and_individual_object_views": True,
            "pair_neutral_rgb": list(pair_neutral),
            "pair_crop_padding_pixels": pair_padding,
            "object_neutral_rgb": list(object_neutral),
            "object_crop_padding_pixels": object_padding,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--feature-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.feature_output, args.manifest_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = extract(config, args.clip, args.feature_output)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"events": result["events"], "output": str(args.feature_output)}))


if __name__ == "__main__":
    main()
