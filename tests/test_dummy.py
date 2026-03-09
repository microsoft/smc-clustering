# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Sanity checks for package imports.

This test module keeps a minimal smoke test in place so the package import path is exercised during test runs.
"""

from diffusion_linking import *


def test_import() -> None:
    """A dummy test to check if the package can be imported."""
    assert True
