from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ContactEpisode:
    object_a: int
    object_b: int
    start_frame: int
    end_frame: int
    contact_rows: int
    impulse: np.ndarray


def contact_episodes(
    collisions: Mapping[str, np.ndarray],
    *,
    step_rate: float = 240.0,
    background_id: int = 65535,
) -> list[ContactEpisode]:
    """Merge step-level MOVi contact rows into object-pair episodes."""

    instances = np.asarray(collisions["instances"], dtype=np.int64)
    frames = np.asarray(collisions["frame"], dtype=np.int64)
    forces = np.asarray(collisions["force"], dtype=np.float64)
    normals = np.asarray(collisions["contact_normal"], dtype=np.float64)
    count = len(frames)
    if instances.shape != (count, 2) or normals.shape != (count, 3):
        raise ValueError("invalid MOVi collision arrays")
    if forces.shape != (count,) or step_rate <= 0:
        raise ValueError("invalid MOVi force log")
    if not all(np.all(np.isfinite(value)) for value in (forces, normals)):
        raise ValueError("collision log contains non-finite values")
    if np.any(frames < 0) or np.any(forces < 0):
        raise ValueError("collision log contains an invalid frame or force")

    rows_by_pair: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for row, ((first, second), frame, force, normal) in enumerate(
        zip(instances, frames, forces, normals, strict=True)
    ):
        if first == background_id or second == background_id or first < 0 or second < 0:
            continue
        object_a, object_b = sorted((int(first), int(second)))
        # MOVi lists body B first; PyBullet's normal points from B towards A.
        # Canonicalize the vector from the lower instance index to the higher one.
        direction = normal if int(first) == object_a else -normal
        vector = float(force) * direction / float(step_rate)
        rows_by_pair.setdefault((object_a, object_b), []).append((row, int(frame), vector))

    episodes: list[ContactEpisode] = []
    for (object_a, object_b), rows in rows_by_pair.items():
        pair_frames = np.asarray([item[1] for item in rows], dtype=np.int64)
        if np.any(np.diff(pair_frames) < 0):
            raise ValueError("collision rows are not chronological")
        current: list[tuple[int, int, np.ndarray]] = []
        previous_frame: int | None = None
        for item in rows:
            frame = item[1]
            if current and previous_frame is not None and frame - previous_frame > 1:
                episodes.append(_make_episode(object_a, object_b, current))
                current = []
            current.append(item)
            previous_frame = frame
        if current:
            episodes.append(_make_episode(object_a, object_b, current))
    return sorted(episodes, key=lambda item: (item.start_frame, item.object_a, item.object_b))


def _make_episode(
    object_a: int,
    object_b: int,
    rows: list[tuple[int, int, np.ndarray]],
) -> ContactEpisode:
    return ContactEpisode(
        object_a=object_a,
        object_b=object_b,
        start_frame=min(item[1] for item in rows),
        end_frame=max(item[1] for item in rows),
        contact_rows=len(rows),
        impulse=np.sum([item[2] for item in rows], axis=0),
    )


def select_isolated_episode(
    episodes: list[ContactEpisode],
    visibility: np.ndarray,
    *,
    lookback: int = 2,
    minimum_visibility: int = 64,
    isolation_window: int = 2,
) -> ContactEpisode | None:
    """Return the first visible contact whose two objects are locally isolated."""

    if lookback < 0:
        raise ValueError("lookback must be nonnegative")
    return select_isolated_episode_at_offsets(
        episodes,
        visibility,
        input_offsets=tuple(range(-lookback, 1)),
        minimum_visibility=minimum_visibility,
        isolation_window=isolation_window,
    )


def select_isolated_episode_at_offsets(
    episodes: list[ContactEpisode],
    visibility: np.ndarray,
    *,
    input_offsets: tuple[int, ...],
    minimum_visibility: int = 64,
    isolation_window: int = 2,
) -> ContactEpisode | None:
    """Return the first isolated contact visible at every requested relative frame."""

    pixels = np.asarray(visibility)
    if pixels.ndim != 2:
        raise ValueError("visibility must have shape objects-by-frames")
    if not input_offsets or len(set(input_offsets)) != len(input_offsets):
        raise ValueError("input offsets must be nonempty and unique")
    offsets = np.asarray(input_offsets, dtype=np.int64)
    for candidate in episodes:
        start = candidate.start_frame
        frames = start + offsets
        if np.any(frames < 0) or np.any(frames >= pixels.shape[1]):
            continue
        if candidate.object_b >= pixels.shape[0]:
            continue
        objects = [candidate.object_a, candidate.object_b]
        if np.any(pixels[np.ix_(objects, frames)] < minimum_visibility):
            continue
        members = {candidate.object_a, candidate.object_b}
        overlaps = [
            other
            for other in episodes
            if other is not candidate
            and members.intersection((other.object_a, other.object_b))
            and abs(other.start_frame - start) <= isolation_window
        ]
        if not overlaps:
            return candidate
    return None


def view_from_world_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return Kubric world vectors in right-down-forward camera coordinates."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("camera quaternion must be a finite wxyz vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("camera quaternion has zero norm")
    w, x, y, z = quaternion / norm
    camera_to_world = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    right_down_forward = np.diag([1.0, -1.0, -1.0])
    return right_down_forward @ camera_to_world.T
