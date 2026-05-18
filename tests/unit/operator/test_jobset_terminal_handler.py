# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the watch-driven JobSet terminal-condition handler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pytest import param

from aiperf.kubernetes.constants import AIPerfLabels, Annotations
from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION
from aiperf.operator.handlers.jobset_terminal import handle_jobset_conditions


def _aiperfjob_body(*, annotations: dict[str, str] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": "ajob", "uid": "uid-ajob"}
    if annotations is not None:
        metadata["annotations"] = annotations
    return {"metadata": metadata, "status": {}}


def _trusted_jobset_body() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "aiperf-ajob",
            "labels": {
                AIPerfLabels.APP_KEY: AIPerfLabels.APP_VALUE,
                AIPerfLabels.JOB_ID: "ajob",
            },
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "ajob",
                    "uid": "uid-ajob",
                }
            ],
        }
    }


@pytest.mark.asyncio
async def test_completed_condition_triggers_annotation_patch() -> None:
    """When the JobSet flips to type=Completed/status=True, set BENCHMARK_COMPLETE on the parent."""
    new_conditions = [
        {"type": "Completed", "status": "True", "reason": "AllJobsCompleted"},
    ]
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(return_value=_aiperfjob_body(annotations={})),
        ),
        patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter,
    ):
        await handle_jobset_conditions(
            old=[],
            new=new_conditions,
            namespace="ns",
            jobset_name="aiperf-ajob",
            jobset_body=_trusted_jobset_body(),
        )
    setter.assert_awaited_once_with("ns", "ajob")


@pytest.mark.asyncio
async def test_non_terminal_condition_change_does_nothing() -> None:
    """A non-terminal condition (Suspended) is a no-op for this handler."""
    new = [{"type": "Suspended", "status": "True"}]
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter,
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(),
        ) as lookup,
    ):
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
        )
    setter.assert_not_awaited()
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_false_status_does_nothing() -> None:
    """A Completed condition with status=False is not terminal-success."""
    new = [{"type": "Completed", "status": "False"}]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter:
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
        )
    setter.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_condition_is_no_op() -> None:
    """type=Failed/status=True stays on the existing monitor-timer recovery path."""
    new = [{"type": "Failed", "status": "True", "reason": "ControllerCrashed"}]
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter,
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(),
        ) as lookup,
    ):
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
        )
    setter.assert_not_awaited()
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_completed_in_old_conditions_skips() -> None:
    """Re-firing on the same Completed condition list is a no-op (saves a CR get)."""
    completed = [{"type": "Completed", "status": "True"}]
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(),
        ) as lookup,
        patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter,
    ):
        await handle_jobset_conditions(
            old=completed,
            new=completed,
            namespace="ns",
            jobset_name="aiperf-ajob",
        )
    lookup.assert_not_awaited()
    setter.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_annotation_skips_redundant_patch() -> None:
    """If the controller pod already set BENCHMARK_COMPLETE, skip the redundant patch."""
    new = [{"type": "Completed", "status": "True"}]
    body = _aiperfjob_body(annotations={Annotations.BENCHMARK_COMPLETE: "true"})
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(return_value=body),
        ),
        patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter,
    ):
        await handle_jobset_conditions(
            old=[],
            new=new,
            namespace="ns",
            jobset_name="aiperf-ajob",
            jobset_body=_trusted_jobset_body(),
        )
    setter.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_owned_jobset_skips_silently() -> None:
    """Sweep-owned JobSets resolve to a non-existent AIPerfJob CR and skip."""
    new = [{"type": "Completed", "status": "True"}]
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter,
    ):
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-someweep"
        )
    setter.assert_not_awaited()


@pytest.mark.asyncio
async def test_jobset_name_without_aiperf_prefix_skips() -> None:
    """A JobSet whose name doesn't start with 'aiperf-' is not ours."""
    new = [{"type": "Completed", "status": "True"}]
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter,
    ):
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="some-other-jobset"
        )
    setter.assert_not_awaited()


# =============================================================================
# Adversarial tests — production-hostile inputs
# =============================================================================


class TestJobsetTerminalAdversarial:
    """Adversarial coverage for ``handle_jobset_conditions`` and
    ``_has_completed_condition``.

    These probe the production-hostile shapes kopf can deliver as the
    JobSet conditions list mutates: None / non-dict entries, missing or
    lowercased status fields, race-with-controller-pod annotations, and
    boundary cases on the ``aiperf-`` prefix-strip.
    """

    @pytest.mark.asyncio
    async def test_old_and_new_both_none_is_no_op(self) -> None:
        """Both `old` and `new` None → no-op; no apiserver work."""
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(),
            ) as lookup,
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await handle_jobset_conditions(
                old=None, new=None, namespace="ns", jobset_name="aiperf-x"
            )
        lookup.assert_not_awaited()
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_completed_and_failed_both_true_completed_wins(self) -> None:
        """If both Completed/True AND Failed/True appear, ``_has_completed_condition``
        returns True (only Completed is checked) → patch fires.

        This is the documented behavior — Failed-condition handling lives on
        the monitor-timer path, and a JobSet shouldn't legitimately have
        both, but defensively we prefer the Completed signal here.
        """
        new = [
            {"type": "Failed", "status": "True"},
            {"type": "Completed", "status": "True"},
        ]
        body = _aiperfjob_body(annotations={})
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=body),
            ),
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await handle_jobset_conditions(
                old=[],
                new=new,
                namespace="ns",
                jobset_name="aiperf-ajob",
                jobset_body=_trusted_jobset_body(),
            )
        setter.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "junk_entry",
        [
            param(None, id="none_entry"),
            param("not-a-dict", id="string_entry"),
            param(42, id="int_entry"),
        ],
    )  # fmt: skip
    async def test_non_dict_condition_entry_is_skipped_not_crashed(
        self, junk_entry: Any
    ) -> None:
        """Defensive: a malformed JobSet status with a non-dict entry in
        the conditions list must be skipped, not crash.

        REGRESSION GUARD: previously ``_has_completed_condition`` called
        ``cond.get("type")`` on each entry, which raises AttributeError
        for None / strings / numbers. Fixed by ``isinstance(cond, dict)``.
        """
        new = [
            junk_entry,
            {"type": "Completed", "status": "True"},
        ]
        body = _aiperfjob_body(annotations={})
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=body),
            ),
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await handle_jobset_conditions(
                old=[],
                new=new,
                namespace="ns",
                jobset_name="aiperf-ajob",
                jobset_body=_trusted_jobset_body(),
            )
        setter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_only_non_dict_entries_treated_as_no_terminal(self) -> None:
        """A list of all junk entries → no Completed → no-op."""
        new = [None, "junk", 7]
        with patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(),
        ) as lookup:
            await handle_jobset_conditions(
                old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
            )
        lookup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_completed_missing_status_field_no_op(self) -> None:
        """type=Completed but no status key → ``.get("status") == "True"`` is
        False → no-op."""
        new = [{"type": "Completed"}]
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(),
            ) as lookup,
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await handle_jobset_conditions(
                old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
            )
        lookup.assert_not_awaited()
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lowercase_true_status_is_no_op(self) -> None:
        """k8s convention: ``status: "True"`` capitalized. Lowercase
        ``"true"`` is non-conformant and treated as not-yet-terminal.

        Pinned strict-equality semantics; if upstream JobSet ever changes
        we revisit here. Better to noop than to spuriously annotate.
        """
        new = [{"type": "Completed", "status": "true"}]
        with patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter:
            await handle_jobset_conditions(
                old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
            )
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_aiperfjob_body_with_no_annotations_key_handled(self) -> None:
        """``metadata.annotations`` absent: existing-annotation check must
        treat that as "no prior annotation" and proceed to patch."""
        new = [{"type": "Completed", "status": "True"}]
        # Note: metadata has no "annotations" key at all.
        body = _aiperfjob_body()
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=body),
            ),
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await handle_jobset_conditions(
                old=[],
                new=new,
                namespace="ns",
                jobset_name="aiperf-ajob",
                jobset_body=_trusted_jobset_body(),
            )
        setter.assert_awaited_once_with("ns", "ajob")

    @pytest.mark.asyncio
    async def test_aiperfjob_body_with_none_annotations_handled(self) -> None:
        """``metadata.annotations = None`` (apiserver merge-patch artifact):
        existing-annotation check must coerce to {} and proceed to patch."""
        new = [{"type": "Completed", "status": "True"}]
        body = _aiperfjob_body()
        body["metadata"]["annotations"] = None
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=body),
            ),
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await handle_jobset_conditions(
                old=[],
                new=new,
                namespace="ns",
                jobset_name="aiperf-ajob",
                jobset_body=_trusted_jobset_body(),
            )
        setter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aiperfjob_body_with_no_metadata_handled(self) -> None:
        """An AIPerfJob body with no metadata cannot prove ownership and skips."""
        new = [{"type": "Completed", "status": "True"}]
        body: dict[str, Any] = {"status": {}}  # NO metadata
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=body),
            ),
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await handle_jobset_conditions(
                old=[],
                new=new,
                namespace="ns",
                jobset_name="aiperf-ajob",
                jobset_body=_trusted_jobset_body(),
            )
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_set_annotation_swallows_apiexception_404(self) -> None:
        """If the AIPerfJob CR was deleted between lookup and patch, the
        404 ApiException is swallowed inside ``_set_benchmark_complete_annotation``
        — handler returns silently, no kopf retry storm."""
        from kubernetes_asyncio.client.exceptions import ApiException

        from aiperf.operator.handlers.jobset_terminal import (
            _set_benchmark_complete_annotation,
        )

        # Mock the k8s client + custom api so the patch raises 404.
        class _ApiCtx:
            async def __aenter__(self) -> Any:
                return object()

            async def __aexit__(self, *_a: Any) -> None:
                return None

        async def mock_patch_fn(*_a: Any, **_kw: Any) -> None:
            raise ApiException(status=404, reason="Not Found")

        from unittest.mock import MagicMock

        custom_obj = MagicMock()
        custom_obj.patch_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=404, reason="Not Found")
        )

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                new=lambda: _ApiCtx(),
            ),
            patch(
                "kubernetes_asyncio.client.CustomObjectsApi",
                return_value=custom_obj,
            ),
        ):
            # Should not raise.
            await _set_benchmark_complete_annotation("ns", "ajob")

    @pytest.mark.asyncio
    async def test_jobset_name_exactly_aiperf_dash_results_in_empty_lookup(
        self,
    ) -> None:
        """``jobset_name == "aiperf-"`` → ``removeprefix`` yields "" →
        AIPerfJob name "" → lookup is called and returns None (since "" is
        not a valid CR name and apiserver returns 404). Pin: no crash."""
        new = [{"type": "Completed", "status": "True"}]
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=None),
            ) as lookup,
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await handle_jobset_conditions(
                old=[], new=new, namespace="ns", jobset_name="aiperf-"
            )
        # lookup is called with the empty-string name — pin behavior.
        lookup.assert_awaited_once()
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_jobset_name_aiperf_no_dash_skips_at_lookup(self) -> None:
        """``jobset_name == "aiperf"`` (no dash) → ``startswith("aiperf-")``
        is False → ``_lookup_aiperfjob_body`` returns None → handler skips."""
        new = [{"type": "Completed", "status": "True"}]
        # Use the real lookup helper here (no mock) to verify the prefix check.
        with patch(
            "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
            new=AsyncMock(),
        ) as setter:
            await handle_jobset_conditions(
                old=[], new=new, namespace="ns", jobset_name="aiperf"
            )
        setter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_fires_double_patch_idempotent(self) -> None:
        """Two concurrent fires for the same JobSet currently both
        patch (idempotent: same key=value). Pin the at-most-twice
        semantic — no crash, no spurious phase change.
        """
        import asyncio

        new = [{"type": "Completed", "status": "True"}]
        body = _aiperfjob_body(annotations={})
        with (
            patch(
                "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
                new=AsyncMock(return_value=body),
            ),
            patch(
                "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
                new=AsyncMock(),
            ) as setter,
        ):
            await asyncio.gather(
                handle_jobset_conditions(
                    old=[],
                    new=new,
                    namespace="ns",
                    jobset_name="aiperf-ajob",
                    jobset_body=_trusted_jobset_body(),
                ),
                handle_jobset_conditions(
                    old=[],
                    new=new,
                    namespace="ns",
                    jobset_name="aiperf-ajob",
                    jobset_body=_trusted_jobset_body(),
                ),
            )
        # Both pass the existing-annotation check (still empty in mocked
        # body) and both call the setter. Idempotent at the apiserver level.
        assert setter.await_count == 2
