# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the namespace-scoped Archive view (``/ns/:ns/archive``).

The Archive view was rewritten as a flat per-namespace list — the route
itself is the namespace filter, so there are no cross-namespace grouping
headers and no "All namespaces" toggle. Each row's test-id is
``arch-row-{namespace}-{name}``; the search input narrows the rows
within the current namespace.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import ArchivePage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_renders_only_namespace_jobs(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/ns/aiperf-bench/archive`` shows only ``aiperf-bench`` rows."""
    p = ArchivePage(
        page=page,
        base_url=live_operator_app.base_url,
        namespace="aiperf-bench",
    )
    await p.goto()
    rows = page.locator("[data-testid^='arch-row-aiperf-bench-']")
    await expect(rows.first).to_be_visible()
    assert await rows.count() > 0
    other = page.locator("[data-testid^='arch-row-ml-lab-']")
    assert await other.count() == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_other_namespace_has_its_own_jobs(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/ns/ml-lab/archive`` shows only ``ml-lab`` rows."""
    p = ArchivePage(
        page=page,
        base_url=live_operator_app.base_url,
        namespace="ml-lab",
    )
    await p.goto()
    rows = page.locator("[data-testid^='arch-row-ml-lab-']")
    await expect(rows.first).to_be_visible()
    assert await rows.count() > 0


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_search_filters_namespace_rows(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The ``arch-search`` input filters rows within the current namespace."""
    p = ArchivePage(
        page=page,
        base_url=live_operator_app.base_url,
        namespace="aiperf-bench",
    )
    await p.goto()
    rows = page.locator("[data-testid^='arch-row-aiperf-bench-']")
    await expect(rows.first).to_be_visible()
    first_id = await rows.first.get_attribute("data-testid")
    name = first_id.removeprefix("arch-row-aiperf-bench-")
    await p.search().fill(name[:6])
    await expect(rows.first).to_be_visible()
