# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""v2 scenario hook + the underscore explicit-set flags it depends on.

REBASED from the v1 ``UserConfig`` ``--scenario`` / ``--unsafe-override`` hook
suite (Task 9). On v1 the model auto-ran ``validate_scenario`` during
construction and stashed ``cfg._scenario_outcome``. On v2 the hook is
``apply_scenario(run)`` (invoked by the resolver chain, not by
``BenchmarkConfig`` construction) and the result lives on
``run.resolved.scenario_outcome``.

RECONCILIATION
--------------
The v1 scenario-lock / unsafe-override / no-op behaviors are already covered,
faithfully and against real models, by
``tests/unit/common/scenario/test_scenario_validator.py``:

- ``test_no_scenario_returns_noop``                  (was: skips-validator-when-absent)
- ``test_outcome_stored_on_resolved``                (was: calls-validator-when-set)
- ``test_wrong_public_loader_raises`` /
  ``_apply_require_loader`` coverage                 (was: lock-error-without-unsafe-override)
- ``test_unsafe_override_converts_errors_to_warnings`` (was: unsafe-override-downgrades)

Those v1 tests are therefore DROPPED here (dup) rather than re-implemented. What
remains in this file is the NET-NEW coverage the sibling does not assert: the
explicit-set underscore flags the scenario validator reads, which moved to the
v2 dataset / phase configs.
"""

from __future__ import annotations

from aiperf.common.enums import CacheBustTarget
from aiperf.config.dataset.config import FileDataset, PublicDataset
from aiperf.config.dataset.content import CacheBustConfig

# ---------------------------------------------------------------------------
# Scenario hook behavior (lock raise / unsafe-override downgrade / no-op) is
# NOT re-tested here: see tests/unit/common/scenario/test_scenario_validator.py
# which builds real BenchmarkRun objects and exercises apply_scenario directly.
# Duplicating it against a synthetic UserConfig stand-in would add no coverage.
# ---------------------------------------------------------------------------


class TestExplicitlySetFlags:
    """The underscore flags ``apply_scenario`` defensively reads.

    The scenario validator distinguishes "user explicitly set X to a
    non-required value" (raise) from "X is at default; auto-fill from the
    scenario spec" (info log). That distinction relies on the underscore
    ``_*_explicitly_set`` flags below; ``test_scenario_validator.py`` proves
    they are wired (e.g. ``test_explicit_no_streaming_raises`` reads
    ``endpoint._streaming_explicitly_set``). These tests pin the flags on their
    v2 model homes so a future refactor that drops them fails loudly here rather
    than silently weakening the scenario lock.
    """

    # -- cache_bust target (moved to aiperf.config.dataset.content) ----------

    def test_cache_bust_target_explicit_flag_when_passed(self) -> None:
        cfg = CacheBustConfig(target=CacheBustTarget.SYSTEM_PREFIX)
        assert cfg._target_explicitly_set is True

    def test_cache_bust_target_explicit_flag_when_omitted(self) -> None:
        cfg = CacheBustConfig()
        assert cfg._target_explicitly_set is False
        assert cfg.target == CacheBustTarget.NONE

    # -- use_think_time_only (moved to FileDataset / PublicDataset) ----------

    def test_use_think_time_only_explicit_flag_when_passed_file_dataset(self) -> None:
        cfg = FileDataset(
            name="main", type="file", path="/fake/trace.jsonl", use_think_time_only=True
        )
        assert cfg._use_think_time_only_explicitly_set is True

    def test_use_think_time_only_explicit_flag_when_omitted_file_dataset(self) -> None:
        cfg = FileDataset(name="main", type="file", path="/fake/trace.jsonl")
        assert cfg._use_think_time_only_explicitly_set is False

    def test_use_think_time_only_explicit_flag_when_passed_public_dataset(self) -> None:
        cfg = PublicDataset(
            name="main", type="public", dataset="sharegpt", use_think_time_only=True
        )
        assert cfg._use_think_time_only_explicitly_set is True

    def test_use_think_time_only_explicit_flag_when_omitted_public_dataset(
        self,
    ) -> None:
        cfg = PublicDataset(name="main", type="public", dataset="sharegpt")
        assert cfg._use_think_time_only_explicitly_set is False


# ---------------------------------------------------------------------------
# DROPPED v1 tests (no v2 home; v1 config-internal that v2 reorganized):
#
# - test_inter_turn_delay_cap_explicit_flag_when_passed/omitted: v2 has the
#   ``inter_turn_delay_cap_seconds`` field on FileDataset/PublicDataset but no
#   ``_inter_turn_delay_cap_explicitly_set`` underscore flag -- the scenario
#   validator never reads one (no inter-turn-delay lock). v1-internal, dropped.
# - test_trace_idle_gap_cap_explicit_flag_when_passed/omitted: same -- v2 keeps
#   the ``trace_idle_gap_cap_seconds`` field but has no explicit-set flag; the
#   trace-idle-gap lock keys off the field VALUE (see scenario validator
#   ``_apply_trace_idle_gap_cap`` and its sibling test
#   ``test_trace_idle_gap_cap_explicit_other_value_raises``), not a flag.
# - test_extra_inputs_parsed_*: v2 has no ``extra_inputs_parsed`` property; the
#   v1 InputConfig canonicalization moved into the CLI->YAML converter
#   (``aiperf.config.flags``) and endpoint ``extra`` is a plain dict. v1-internal.
# - test_detected_loader_default_none: v2 has no ``detected_loader`` attribute;
#   loader identity is derived on demand by the scenario validator's
#   ``_detect_loader(run)`` from the dataset config. v1-internal, dropped.
# ---------------------------------------------------------------------------
