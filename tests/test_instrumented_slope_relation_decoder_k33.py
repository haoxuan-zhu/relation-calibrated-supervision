from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_instrumented_slope_geometry_k28 as k28
import causalverse_slope_preflight as slope
import run_instrumented_slope_relation_decoder_k33 as k33


PARAMETERS = slope.RelationParameters(0.9, 0.1, 0.5, 0.1, 9.81, 19.62)


def make_model(seed: int = 7) -> k33.InstrumentedSlopeRelationDecoder:
    slope.set_global_seed(seed)
    return k33.InstrumentedSlopeRelationDecoder(
        8,
        [12, 6],
        PARAMETERS,
        0.4,
        np.zeros(5),
        np.ones(5),
        np.asarray([35.0, 10.0]),
        np.asarray([10.0, 2.0]),
        [0.0, 1.0],
    )


def test_zero_initialized_decoder_starts_at_mean_roughness() -> None:
    model = make_model()
    features = torch.randn(5, 8)
    side = torch.tensor([[35.0, 10.0]]).expand(5, -1)
    _, parts = model.components(features, side)
    assert torch.allclose(parts["roughness"], torch.full((5,), 0.4), atol=1.0e-6)


def test_decoder_matches_numpy_physical_equations() -> None:
    model = make_model()
    roughness = torch.tensor([0.1, 0.5, 0.9])
    side = torch.tensor([[20.0, 9.0], [35.0, 10.0], [50.0, 11.0]])
    actual = model.physical_decode(roughness, side).detach().numpy()
    expected = k28.physical_state_from_roughness(
        side.numpy(), roughness.numpy(), PARAMETERS
    )
    assert np.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_roughness_parameterization_stays_inside_physical_interval() -> None:
    model = make_model()
    with torch.no_grad():
        model.output.bias.fill_(100.0)
    features = torch.zeros(4, 8)
    side = torch.tensor([[35.0, 10.0]]).expand(4, -1)
    _, high = model.components(features, side)
    with torch.no_grad():
        model.output.bias.fill_(-100.0)
    _, low = model.components(features, side)
    assert torch.all(high["roughness"] <= 1.0)
    assert torch.all(low["roughness"] >= 0.0)


def test_matched_conditions_share_initial_state_and_parameter_count() -> None:
    first = make_model(9)
    second = make_model(9)
    assert slope.state_dict_sha256(first.state_dict()) == slope.state_dict_sha256(
        second.state_dict()
    )
    assert sum(parameter.numel() for parameter in first.parameters()) == sum(
        parameter.numel() for parameter in second.parameters()
    )
