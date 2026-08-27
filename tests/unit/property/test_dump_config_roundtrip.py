# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dump_config -> load_config_from_string round-trip for every bundled template.

If the user passes ``--dump-config`` (or programmatically calls
``aiperf.config.dump_config(cfg)``) and then re-loads the result, the
re-loaded config must equal the original. This protects against:

- Field aliases that don't round-trip (alias on dump, name on reload).
- Custom ``BeforeValidator``/``AfterValidator`` chains that mutate values
  on load but not on dump (or vice versa).
- ``mode="json"`` serialization that drops/coerces fields.
- Sweep envelope keys that ``model_dump`` flattens incorrectly.

The round-trip is the canonical "dumped output is a valid input" contract.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import pytest

from aiperf.config.loader.core import (
    dump_config,
    load_config,
    load_config_from_string,
)

TEMPLATES_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "config"
    / "templates"
)

_TEMPLATE_ENV_DEFAULTS = {
    "MODEL_NAME": "meta-llama/Llama-3.1-8B-Instruct",
    "INFERENCE_URL": "http://localhost:8000/v1/chat/completions",
    "METRICS_URL": "http://localhost:8000/metrics",
    "TIMEOUT": "600.0",
    "BENCHMARK_SEED": "42",
    "DURATION": "300",
    "TARGET_RATE": "30.0",
    "MAX_CONCURRENCY": "64",
    "NUM_RUNS": "3",
    "COOLDOWN": "30",
    "ARTIFACTS_DIR": "./artifacts/test",
}


@pytest.fixture(autouse=True)
def _set_template_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _TEMPLATE_ENV_DEFAULTS.items():
        monkeypatch.setenv(key, os.environ.get(key, value))


def _normalize(obj: Any) -> Any:
    """Drop volatile fields that legitimately don't round-trip.

    Currently empty -- if a future field is intentionally non-roundtrippable
    (e.g. ephemeral CLI-runtime overrides), normalize it here with a comment.
    """
    return obj


@pytest.mark.parametrize(
    "template_path",
    sorted(TEMPLATES_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_dump_config_roundtrip(template_path: pathlib.Path) -> None:
    """Round-trip every bundled template through ``dump_config`` + reload.

    The reloaded config must be structurally equal to the original. Failure
    means ``dump_config`` produced YAML that the loader cannot reconstruct
    -- a silent contract violation for users who use ``--dump-config`` to
    snapshot a resolved config.
    """
    original = load_config(template_path)
    dumped = dump_config(original)
    assert isinstance(dumped, str) and dumped.strip(), (
        "dump_config returned empty string"
    )
    reloaded = load_config_from_string(dumped)
    # Normalize before compare so ephemeral fields don't trip the test.
    a = _normalize(original.model_dump(mode="json", by_alias=True))
    b = _normalize(reloaded.model_dump(mode="json", by_alias=True))
    if a != b:
        # Surface a compact diff hint instead of a 1000-line dict dump.
        diff_keys = sorted(set(a) ^ set(b))
        common_diff = {
            k: (a.get(k), b.get(k)) for k in (set(a) & set(b)) if a.get(k) != b.get(k)
        }
        raise AssertionError(
            f"dump_config round-trip mismatch for {template_path.name}.\n"
            f"  Keys only on one side: {diff_keys}\n"
            f"  Differing common keys: {sorted(common_diff)}\n"
            f"  Sample: {{k: (orig, reloaded)}} = "
            f"{ {k: common_diff[k] for k in list(common_diff)[:3]} }"
        )


def test_dump_config_roundtrip_directory_is_non_empty() -> None:
    """Guard against an accidental empty glob silently passing the test."""
    assert sorted(TEMPLATES_DIR.glob("*.yaml")), (
        f"No templates discovered under {TEMPLATES_DIR}"
    )
