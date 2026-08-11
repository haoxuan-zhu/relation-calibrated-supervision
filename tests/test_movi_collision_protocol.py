from __future__ import annotations

import numpy as np
import pytest

from relation_tube.movi import (
    contact_episodes,
    select_isolated_episode,
    select_isolated_episode_at_offsets,
    view_from_world_matrix,
)


def test_contact_rows_are_merged_and_reoriented() -> None:
    collisions = {
        "instances": np.array([[2, 1], [1, 2], [2, 1], [4, 3]]),
        "frame": np.array([5, 5, 6, 10]),
        "force": np.array([24.0, 24.0, 48.0, 12.0]),
        "contact_normal": np.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
    }
    episodes = contact_episodes(collisions, step_rate=24.0)

    assert [(item.object_a, item.object_b) for item in episodes] == [(1, 2), (3, 4)]
    assert episodes[0].contact_rows == 3
    np.testing.assert_allclose(episodes[0].impulse, [-4.0, 0.0, 0.0])
    np.testing.assert_allclose(episodes[1].impulse, [0.0, -0.5, 0.0])


def test_background_rows_and_separate_contacts_are_not_counted_together() -> None:
    collisions = {
        "instances": np.array([[0, 65535], [0, 1], [0, 1]]),
        "frame": np.array([3, 4, 8]),
        "force": np.ones(3),
        "contact_normal": np.tile([1.0, 0.0, 0.0], (3, 1)),
    }
    episodes = contact_episodes(collisions)
    assert [(item.start_frame, item.end_frame) for item in episodes] == [(4, 4), (8, 8)]


def test_episode_selection_uses_visibility_and_pair_isolation() -> None:
    collisions = {
        "instances": np.array([[0, 1], [0, 2], [2, 3]]),
        "frame": np.array([3, 4, 9]),
        "force": np.ones(3),
        "contact_normal": np.tile([1.0, 0.0, 0.0], (3, 1)),
    }
    episodes = contact_episodes(collisions)
    visibility = np.full((4, 12), 100)
    selected = select_isolated_episode(episodes, visibility, isolation_window=2)
    assert selected is not None
    assert (selected.object_a, selected.object_b, selected.start_frame) == (2, 3, 9)


def test_offset_selection_checks_exact_input_frames() -> None:
    collisions = {
        "instances": np.array([[0, 1], [2, 3]]),
        "frame": np.array([4, 8]),
        "force": np.ones(2),
        "contact_normal": np.tile([1.0, 0.0, 0.0], (2, 1)),
    }
    episodes = contact_episodes(collisions)
    visibility = np.full((4, 12), 100)
    visibility[1, 1] = 0
    selected = select_isolated_episode_at_offsets(
        episodes, visibility, input_offsets=(-3, -2, -1)
    )
    assert selected is not None
    assert (selected.object_a, selected.object_b, selected.start_frame) == (2, 3, 8)


def test_nonchronological_pair_rows_are_rejected() -> None:
    collisions = {
        "instances": np.array([[0, 1], [0, 1]]),
        "frame": np.array([5, 4]),
        "force": np.ones(2),
        "contact_normal": np.tile([1.0, 0.0, 0.0], (2, 1)),
    }
    with pytest.raises(ValueError, match="chronological"):
        contact_episodes(collisions)


def test_identity_camera_uses_right_down_forward_axes() -> None:
    transform = view_from_world_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(transform, np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_allclose(transform @ transform.T, np.eye(3), atol=1.0e-12)
