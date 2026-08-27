# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip regression: every bundled template loads -> dumps -> reloads.

Covers R2-H2 from round-2 adversarial review: ``GridSweep.type="grid"`` is
a default Field, so ``model_dump(exclude_defaults=True)`` strips the
discriminator and reload fails with ``union_tag_not_found``. The fix in
``dump_config`` re-injects ``sweep.type`` so the round-trip is stable for
every template, not just sweep templates.

Failure mode this test locks in: any new template (or any future change
that adds a defaulted-discriminator union) that breaks dump -> reload.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from aiperf.config.loader import (
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


@pytest.mark.parametrize(
    "template_path",
    sorted(TEMPLATES_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_dump_reload_roundtrip_for_every_template(
    template_path: pathlib.Path,
) -> None:
    """load -> dump -> reload preserves the config for every bundled template.

    Asserts the reloaded config dumps to the same shape as the first dump
    (a fixed point under dump_config). This catches dropped discriminators,
    runtime-default leakage, and any other shape drift on the round-trip.
    """
    cfg1 = load_config(template_path)
    dumped1 = dump_config(cfg1)
    cfg2 = load_config_from_string(dumped1)
    dumped2 = dump_config(cfg2)
    assert dumped1 == dumped2, (
        f"Round-trip not idempotent for {template_path.name}:\n"
        f"--- first dump ---\n{dumped1}\n--- second dump ---\n{dumped2}"
    )


def test_sweep_discriminator_survives_dump() -> None:
    """The grid sweep ``type:`` discriminator is preserved on dump.

    Targeted regression for R2-H2: ``exclude_defaults=True`` would strip
    ``type: grid`` because it's the default; the discriminated union
    rejects the dumped YAML on reload. ``dump_config`` re-injects the
    discriminator so this scenario round-trips.
    """
    cfg1 = load_config(TEMPLATES_DIR / "speed_bench_sweep.yaml")
    assert cfg1.sweep is not None
    dumped = dump_config(cfg1)
    assert "type: grid" in dumped
    cfg2 = load_config_from_string(dumped)
    assert cfg2.sweep is not None
    assert cfg2.sweep.type == "grid"
