# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discriminated-union narrowing for the ``DatasetConfig`` union.

``BenchmarkConfig.get_default_dataset()`` returns a ``DatasetConfig`` --
``SyntheticDataset | FileDataset | PublicDataset`` -- and several dataset knobs
are declared on only SOME members. Reading them with a per-field
``getattr(dataset, "field", default)`` probe hides which members actually carry
the field and silently invents a default for the ones that do not.

These helpers narrow the union ONCE per call site so the fields can be read
directly. Every helper returns ``None`` for a dataset that does not carry the
group, which is the same "knob unset" outcome the ``getattr`` defaults produced.

They take the dataset OBJECT (not the run) because the call sites hold
different handles -- a ``BenchmarkRun``, a ``BenchmarkConfig``, or a bare
``DatasetConfig`` -- and ``cfg.get_default_dataset()`` is the one expression
they all share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.config.dataset.config import FileDataset, PublicDataset

__all__ = ["as_file_dataset", "as_trace_replay_dataset"]


def as_file_dataset(dataset: object | None) -> FileDataset | None:
    """Return ``dataset`` iff it is a :class:`FileDataset`, else ``None``.

    Narrows for the ``FileDataset``-only knobs: ``path``, ``graph_format``,
    ``replay_speedup``, ``open_loop_replay``, ``open_loop_strict``, and the
    other baseten/graph replay fields. ``SyntheticDataset`` and
    ``PublicDataset`` declare none of these, so they yield ``None``.
    """
    from aiperf.config.dataset.config import FileDataset

    return dataset if isinstance(dataset, FileDataset) else None


def as_trace_replay_dataset(
    dataset: object | None,
) -> FileDataset | PublicDataset | None:
    """Return ``dataset`` iff it carries the shared trace-replay knobs.

    ``synthesis``, ``ignore_trace_delays``, ``use_think_time_only``, and
    ``trace_idle_gap_cap_seconds`` are declared on BOTH ``FileDataset`` and
    ``PublicDataset`` but NOT ``SyntheticDataset``, which yields ``None``.
    """
    from aiperf.config.dataset.config import FileDataset, PublicDataset

    return dataset if isinstance(dataset, FileDataset | PublicDataset) else None
