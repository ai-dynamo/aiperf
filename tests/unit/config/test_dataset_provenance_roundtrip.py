# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance flags must survive the sweep orchestrator's serialization boundary.

The multi-run orchestrator writes each run with
``run.model_dump(mode="json", exclude_none=True)``
(``orchestrator/local_executor.py::_prepare_run_artifacts``) and the subprocess
reads it back with ``BenchmarkRun.model_validate``
(``orchestrator/subprocess_runner.py``). That round trip destroys
``model_fields_set``: every dumped key comes back marked "set". Any dataset
provenance flag derived from ``model_fields_set`` therefore reads as forged
"explicit intent" in the subprocess -- which is the only place the resolver
chain runs (``cli_runner/_single_run.py``).

These tests pin the two flags that were broken this way.
"""

from __future__ import annotations

import orjson
import pytest
from pytest import param

from aiperf.common.enums import DatasetType
from aiperf.config.dataset import FileDataset, PublicDataset
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.plugin.enums import PublicDatasetType


def _round_trip(model):
    """Replay the orchestrator's dump -> validate boundary on one model."""
    payload = orjson.loads(
        orjson.dumps(model.model_dump(mode="json", exclude_none=True))
    )
    return type(model).model_validate(payload)


def _public_cli(**extra) -> CLIConfig:
    return CLIConfig(
        model_names=["test-model"],
        streaming=True,
        public_dataset=PublicDatasetType.SHAREGPT,
        **extra,
        **CLIConfig(concurrency=2).model_dump(exclude_unset=True),
    )


# ---------------------------------------------------------------------------
# entries_explicit
# ---------------------------------------------------------------------------


def test_entries_explicit_false_survives_round_trip() -> None:
    """Converter-pinned ``entries_explicit=False`` must survive serialization.

    ``--num-conversations`` populates ``entries`` as a *fallback*, so the CLI
    converter pins ``_entries_explicit=False``. If that False is dropped by the
    dump, ``_resolve_entries_explicit`` re-promotes it to True in the subprocess
    and ``SemiAnalysisCCTracesWekaLoader`` silently caps the HuggingFace corpus
    to the request-count prefix (``num_entries = cap``).
    """
    ds = convert_cli_to_aiperf(
        _public_cli(conversation_num=10)
    ).benchmark.get_default_dataset()
    assert isinstance(ds, PublicDataset)
    assert ds.entries == 10
    assert ds.entries_explicit is False

    rt = _round_trip(ds)
    assert rt.entries == 10
    assert rt.entries_explicit is False, (
        "round trip forged explicit intent: the Weka loader would cap the "
        "corpus to the --num-conversations/--request-count prefix"
    )

    dumped = ds.model_dump(mode="json", exclude_none=True)
    assert "entries_explicit" in dumped, (
        "entries_explicit must serialize (no exclude=True) or its provenance "
        "cannot cross the orchestrator boundary"
    )


def test_entries_explicit_true_survives_round_trip() -> None:
    """Explicit ``--num-dataset-entries`` intent also survives unchanged."""
    ds = convert_cli_to_aiperf(
        _public_cli(conversation_num_dataset_entries=42)
    ).benchmark.get_default_dataset()
    assert ds.entries_explicit is True
    assert _round_trip(ds).entries_explicit is True


def test_entries_explicit_alias_still_accepted() -> None:
    """The CLI converter's ``_entries_explicit`` sentinel key still validates."""
    d = build_dataset(_public_cli(conversation_num=10))
    assert d["_entries_explicit"] is False
    ds = PublicDataset.model_validate(
        {"name": "default", **d, "type": DatasetType.PUBLIC}
    )
    assert ds.entries_explicit is False


def test_entries_without_sentinel_still_promoted_to_explicit() -> None:
    """A YAML/programmatic ``entries`` (no sentinel) is still explicit intent."""
    ds = PublicDataset.model_validate(
        {
            "name": "default",
            "type": DatasetType.PUBLIC,
            "dataset": PublicDatasetType.SHAREGPT,
            "entries": 7,
        }
    )
    assert ds.entries_explicit is True


# ---------------------------------------------------------------------------
# use_think_time_only
# ---------------------------------------------------------------------------


def _dataset(cls, **extra):
    base = (
        {"type": DatasetType.FILE, "path": "/tmp/trace.jsonl"}
        if cls is FileDataset
        else {"type": DatasetType.PUBLIC, "dataset": PublicDatasetType.SHAREGPT}
    )
    return cls(name="default", **base, **extra)


@pytest.mark.parametrize(
    "cls",
    [param(FileDataset, id="file"), param(PublicDataset, id="public")],
)  # fmt: skip
def test_use_think_time_only_unset_stays_unset_across_round_trip(cls) -> None:
    """Unset must not be forged into explicit intent by the dump.

    ``_use_think_time_only_explicitly_set`` gates scenario auto-fill
    (``common/scenario/validator.py::_apply_use_think_time_only``): unset means
    "the scenario may force it True", explicit False means "the user opted out,
    raise a violation". A forged explicit-False turns every auto-fill into a
    spurious ``--use-think-time-only`` violation.
    """
    ds = _dataset(cls)
    assert ds.use_think_time_only is None
    assert ds._use_think_time_only_explicitly_set is False

    rt = _round_trip(ds)
    assert rt.use_think_time_only is None
    assert rt._use_think_time_only_explicitly_set is False


@pytest.mark.parametrize(
    "cls",
    [param(FileDataset, id="file"), param(PublicDataset, id="public")],
)  # fmt: skip
@pytest.mark.parametrize(
    "value",
    [param(True, id="true"), param(False, id="false")],
)  # fmt: skip
def test_use_think_time_only_authored_value_survives_round_trip(cls, value) -> None:
    """An authored True/False keeps both its value and its explicit provenance."""
    ds = _dataset(cls, use_think_time_only=value)
    assert ds._use_think_time_only_explicitly_set is True

    rt = _round_trip(ds)
    assert rt.use_think_time_only is value
    assert rt._use_think_time_only_explicitly_set is True


def test_scenario_autofill_survives_round_trip() -> None:
    """End-to-end consumer: the scenario validator must still auto-fill.

    Before the fix the round trip produced ``violations=['--use-think-time-only']``
    where a fresh config produced ``applied=['use_think_time_only']``.
    """
    from aiperf.common.scenario.validator import _apply_use_think_time_only

    class _Spec:
        name = "test-scenario"
        require_use_think_time_only = True

    class _Run:
        def __init__(self, ds):
            self.cfg = type("C", (), {"get_default_dataset": lambda _s: ds})()

    ds = _round_trip(_dataset(FileDataset))
    violations: list = []
    applied: list[str] = []
    _apply_use_think_time_only(_Run(ds), _Spec(), violations, applied)

    assert violations == []
    assert applied == ["use_think_time_only"]
    assert ds.use_think_time_only is True


def test_use_think_time_only_exclusivity_still_rejects_true_pair() -> None:
    """The mutual-exclusion check keys off ``is True``, not truthiness."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _dataset(FileDataset, use_think_time_only=True, ignore_trace_delays=True)

    # An explicit False alongside ignore_trace_delays is fine.
    ds = _dataset(FileDataset, use_think_time_only=False, ignore_trace_delays=True)
    assert ds.ignore_trace_delays is True
