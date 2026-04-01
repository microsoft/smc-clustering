# Copyright (c) Lancaster University.
# Licensed under the MIT license.

"""Tests for the sweep_eval_clustering_smc helper utilities."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "sweep_eval_clustering_smc.py"


@pytest.fixture(scope="module")
def sweep_module() -> types.ModuleType:
    """Load the sweep helper module for tests."""
    spec = importlib.util.spec_from_file_location("sweep_eval_clustering_smc", MODULE_PATH)
    if spec is None:
        raise AssertionError("unable to load sweep_eval_clustering_smc module")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_parse_square_list_produces_cartesian_values(sweep_module: types.ModuleType):
    """Run test parse square list produces cartesian values."""
    spec = sweep_module.parse_sweep_spec("[1,2,3]")
    assert spec.values == ["1", "2", "3"]
    assert spec.tied is False


def test_parse_parentheses_marks_tied(sweep_module: types.ModuleType):
    """Run test parse parentheses marks tied."""
    spec = sweep_module.parse_sweep_spec("(10, 20,30)")
    assert spec.values == ["10", "20", "30"]
    assert spec.tied is True


def test_generate_combos_with_tied_parameters(sweep_module: types.ModuleType):
    """Run test generate combos with tied parameters."""
    combos = sweep_module.generate_combos(
        {"seed": ["0", "1"]},
        [
            ("max_particles", ["50", "75"]),
            ("max_evals", ["50", "75"]),
        ],
    )
    assert combos == [
        {"seed": "0", "max_particles": "50", "max_evals": "50"},
        {"seed": "1", "max_particles": "50", "max_evals": "50"},
        {"seed": "0", "max_particles": "75", "max_evals": "75"},
        {"seed": "1", "max_particles": "75", "max_evals": "75"},
    ]


def test_generate_combos_requires_matching_tied_lengths(sweep_module: types.ModuleType):
    """Run test generate combos requires matching tied lengths."""
    with pytest.raises(ValueError, match="Tied parameters must share the same length"):
        sweep_module.generate_combos(
            {},
            [
                ("max_particles", ["10", "20"]),
                ("max_evals", ["30"]),
            ],
        )
