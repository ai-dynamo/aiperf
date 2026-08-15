# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal provenance flags stay out of user-facing exports without losing the sweep round trip.

Three flags record whether the author explicitly chose a value:

    EndpointConfig.streaming_explicitly_set
    CacheBustConfig.target_explicitly_set
    BasePhaseConfig.concurrency_explicitly_set

They are deliberately SERIALIZED and cannot use ``exclude=True``: the sweep
orchestrator round-trips every run through
``local_executor._prepare_run_artifacts`` (``model_dump(exclude_none=True)``) ->
``subprocess_runner`` (``model_validate``), and ``model_fields_set`` is
uninformative on the far side because every dumped key returns marked "set".
Dropping them would let an unset value read as explicitly authored.

Serialized is not the same as user-facing, though. Because the validators
ASSIGN the flag, pydantic adds it to ``model_fields_set``, so the exporters'
``exclude_unset=True`` did not drop it and the keys surfaced in
``profile_export_aiperf.json`` under ``input_config`` -- keys that do not exist
on ``origin/main`` and that downstream config diffing would see as new.

The two dump shapes differ in exactly the way that separates the concerns:

    exporters        model_dump(mode="json", exclude_unset=True, exclude_none=True)
    sweep boundary   model_dump(mode="json", exclude_none=True)

so discarding the flag from ``model_fields_set`` after the validator settles it
hides it from the exporters while leaving the sweep dump untouched. These tests
pin both halves; a regression that restores the leak, or one that breaks
provenance across the round trip, fails here.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.config import AIPerfConfig
from aiperf.config.dataset.content import CacheBustConfig
from aiperf.config.endpoint import EndpointConfig
from aiperf.config.phases import ConcurrencyPhase

# The exporters' dump shape (metrics_json_exporter, timeslice, aggregate
# confidence, server-metrics json/parquet all use these two together).
EXPORT_DUMP = {"mode": "json", "exclude_unset": True, "exclude_none": True}
# local_executor._prepare_run_artifacts -- note: NO exclude_unset.
SWEEP_DUMP = {"mode": "json", "exclude_none": True}

PROVENANCE_FLAGS = (
    "streaming_explicitly_set",
    "target_explicitly_set",
    "concurrency_explicitly_set",
)


def _config(**overrides) -> AIPerfConfig:
    endpoint = {"url": "http://127.0.0.1:1", "type": "chat"}
    endpoint.update(overrides.pop("endpoint", {}))
    phase = {"name": "profiling", "type": "concurrency", "requests": 5}
    phase.update(overrides.pop("phase", {}))
    return AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["m"],
                "endpoint": endpoint,
                "dataset": {
                    "type": "synthetic",
                    "prompts": {"cache_bust": {"target": "first_turn_prefix"}},
                },
                "phases": [phase],
            }
        }
    )


@pytest.mark.parametrize("flag", PROVENANCE_FLAGS)  # fmt: skip
def test_export_dump_omits_every_provenance_flag(flag: str) -> None:
    """The user-facing export shape must not carry internal provenance."""
    payload = str(_config().model_dump(**EXPORT_DUMP))
    assert flag not in payload, (
        f"{flag} leaked into the exporter dump shape; it would appear in "
        "profile_export_aiperf.json under input_config"
    )


@pytest.mark.parametrize("flag", PROVENANCE_FLAGS)  # fmt: skip
def test_sweep_dump_still_carries_every_provenance_flag(flag: str) -> None:
    """The sweep boundary must keep them, or provenance is forged downstream."""
    payload = str(_config().model_dump(**SWEEP_DUMP))
    assert flag in payload, (
        f"{flag} vanished from the sweep dump shape; the subprocess would "
        "re-derive it from model_fields_set and read an unset value as authored"
    )


@pytest.mark.parametrize(
    ("model_cls", "kwargs", "value_field", "flag"),
    [
        param(
            EndpointConfig,
            {"url": "http://127.0.0.1:1", "type": "chat"},
            "streaming",
            "streaming_explicitly_set",
            id="endpoint-streaming",
        ),
        param(
            CacheBustConfig, {}, "target", "target_explicitly_set", id="cache-bust-target"
        ),
        param(
            ConcurrencyPhase,
            {"name": "profiling", "type": "concurrency", "requests": 5},
            "concurrency",
            "concurrency_explicitly_set",
            id="phase-concurrency",
        ),
    ],
)  # fmt: skip
@pytest.mark.parametrize("authored", [False, True], ids=["unset", "authored"])  # fmt: skip
def test_provenance_survives_the_sweep_round_trip(
    model_cls: type,
    kwargs: dict,
    value_field: str,
    flag: str,
    authored: bool,
) -> None:
    """Round-tripping through the sweep boundary preserves the ORIGINAL intent.

    This is the property the serialization exists for: on the private-attr
    design the far side re-derives from ``model_fields_set``, where the dumped
    value key reads as "set" and forges ``True`` for a user who never authored
    it.
    """
    data = dict(kwargs)
    if authored:
        # An explicit value the user really did type.
        data[value_field] = {
            "streaming": False,
            "target": "first_turn_prefix",
            "concurrency": 8,
        }[value_field]

    original = model_cls.model_validate(data)
    assert getattr(original, flag) is authored

    round_tripped = model_cls.model_validate(original.model_dump(**SWEEP_DUMP))
    assert getattr(round_tripped, flag) is authored, (
        "provenance was lost or forged across the sweep dump/validate boundary"
    )
    # ...and the far side still keeps it out of ITS export (a sweep cell writes
    # its own profile_export_aiperf.json).
    assert flag not in str(round_tripped.model_dump(**EXPORT_DUMP))


@pytest.mark.parametrize("authored", [False, True], ids=["unset", "authored"])  # fmt: skip
def test_plan_expansion_dump_rederives_the_truth(authored: bool) -> None:
    """``config/loader/plan.py`` re-expands variations from an exclude_unset dump.

    That path is config RESOLUTION, not export, so it must not be collateral
    damage. It is self-consistent either way: ``exclude_unset=True`` drops the
    value field and the flag TOGETHER, so a re-validated variation derives the
    same answer the original held.
    """
    data = {"url": "http://127.0.0.1:1", "type": "chat"}
    if authored:
        data["streaming"] = False

    original = EndpointConfig.model_validate(data)
    envelope = original.model_dump(mode="python", exclude_none=True, exclude_unset=True)
    variation = EndpointConfig.model_validate(envelope)

    assert variation._streaming_explicitly_set is authored
    assert variation.streaming == original.streaming


def test_backcompat_underscore_properties_still_read_the_flag() -> None:
    """The scenario validator reads these via the ``_``-prefixed properties."""
    cfg = _config(endpoint={"streaming": False}, phase={"concurrency": 8})
    endpoint = cfg.benchmark.endpoint
    cache_bust = cfg.benchmark.get_default_dataset().prompts.cache_bust

    assert endpoint._streaming_explicitly_set is True
    assert cache_bust._target_explicitly_set is True
    assert cfg.benchmark.phases[0]._concurrency_explicitly_set is True
