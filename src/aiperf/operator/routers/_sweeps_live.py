# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-CRD enumeration helpers for the sweeps router.

Live mid-run state lives in the AIPerfJob CRs labelled
``aiperf.nvidia.com/sweep=<sweep>`` — not in the parent AIPerfSweep CR's
``status.aggregate.children``, which the sweep-controller patches only at
aggregation completion. To render meaningful UI before that patch lands,
the route handlers fall back to enumerating the labelled child CRs and
synthesizing the same response shapes the post-aggregation paths emit.

Kept in a sibling to ``sweeps.py`` so that file stays under the 500-line
ergonomics ceiling.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from kubernetes_asyncio import client as k8s
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.endpoint_credentials import (
    redact_sweep_display_label,
    redact_sweep_public_data,
)
from aiperf.operator.routers.sweeps_models import (
    ChildrenManifestEntry,
    ChildrenManifestResponse,
)

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

__all__ = ["children_manifest_from_live_aiperfjobs"]

_TRIAL_SUFFIX_RE = re.compile(r"-t(\d+)$")
_SWEEP_LABEL = "aiperf.nvidia.com/sweep"
_SWEEP_RUN_EPOCH_LABEL = "aiperf.nvidia.com/sweep-run-epoch"
_VARIATION_INDEX_LABEL = "aiperf.nvidia.com/variation-index"
_VARIATION_LABEL_LABEL = "aiperf.nvidia.com/variation-label"
# Values live in an annotation rather than a label because label values are
# limited to 63 characters and may not contain JSON punctuation.
_VARIATION_VALUES_ANNOTATION = "aiperf.nvidia.com/variation-values"


async def children_manifest_from_live_aiperfjobs(
    api: ApiClient,
    namespace: str,
    sweep_name: str,
) -> ChildrenManifestResponse | None:
    """Synthesize a children manifest by listing live AIPerfJob CRs.

    Powers the live-variations rollup card on SweepDetail when the sweep
    is mid-run and ``status.aggregate.children`` is empty. Trial index is
    parsed from the child name's trailing ``-t<N>`` suffix (the only
    place trial index is preserved on the AIPerfJob CR;
    ``aiperf.nvidia.com/trial-index`` is set on the JobSet child Pod, not
    on the parent CR's labels).

    Returns None when the AIPerfJob CRD list call fails or no children
    are labelled — callers fall through to the on-disk archive lookup.
    """
    selector = f"{_SWEEP_LABEL}={sweep_name}"
    try:
        custom = k8s.CustomObjectsApi(api)
        resp = await custom.list_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfjobs",
            label_selector=selector,
        )
    except ApiException:
        return None

    items = resp.get("items") or []
    if not items:
        return None

    sweep_run_epoch: str | None = None
    children: list[ChildrenManifestEntry] = []
    for child in items:
        meta = child.get("metadata") or {}
        labels = meta.get("labels") or {}
        annotations = meta.get("annotations") or {}
        if labels.get(_SWEEP_LABEL) != sweep_name:
            continue
        sweep_run_epoch = sweep_run_epoch or labels.get(_SWEEP_RUN_EPOCH_LABEL)
        cname = meta.get("name") or ""
        m = _TRIAL_SUFFIX_RE.search(cname)
        trial_index = int(m.group(1)) if m else None
        idx = labels.get(_VARIATION_INDEX_LABEL)
        try:
            variation_index = int(idx) if idx is not None else 0
        except (TypeError, ValueError):
            variation_index = 0
        children.append(
            ChildrenManifestEntry(
                namespace=namespace,
                name=cname,
                variation_index=variation_index,
                variation_label=redact_sweep_display_label(
                    labels.get(_VARIATION_LABEL_LABEL) or ""
                ),
                # This route is the ONLY manifest source while a child is still
                # running: status.aggregate.children gains an entry only after
                # the child terminates (k8s_executor._record_terminal_child).
                # Without the annotation read here, a live adaptive sweep shows
                # nothing but `search_iter_NNNN` until it finishes.
                variation_values=str(
                    redact_sweep_public_data(
                        annotations.get(_VARIATION_VALUES_ANNOTATION) or "",
                        path="variation_values",
                    )
                ),
                trial_index=trial_index,
                child_run_epoch=str((child.get("status") or {}).get("runEpoch") or ""),
            )
        )
    if not children:
        return None
    children.sort(key=lambda c: (c.variation_index, c.trial_index or 0))
    return ChildrenManifestResponse(
        sweep_run_epoch=sweep_run_epoch or "",
        children=children,
    )
