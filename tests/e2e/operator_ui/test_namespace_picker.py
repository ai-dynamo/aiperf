# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E coverage for the cross-namespace picker mounted at ``/``."""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import NamespacePickerPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_picker_renders_one_tile_per_namespace(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    p = NamespacePickerPage(page=page, base_url=live_operator_app.base_url)
    await p.goto()
    await expect(p.tile("aiperf-bench")).to_be_visible()
    await expect(p.tile("ml-lab")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_picker_tile_shows_phase_chips(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    p = NamespacePickerPage(page=page, base_url=live_operator_app.base_url)
    await p.goto()
    tile = p.tile("aiperf-bench")
    # At least one phase-chip should be visible inside the tile.
    chips = tile.locator(".np-chip")
    await expect(chips.first).to_be_visible()
    assert await chips.count() >= 1


@pytest.mark.asyncio(loop_scope="session")
async def test_picker_search_filters_tiles(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    p = NamespacePickerPage(page=page, base_url=live_operator_app.base_url)
    await p.goto()
    await p.search().fill("aiperf")
    await expect(p.tile("aiperf-bench")).to_be_visible()
    await expect(p.tile("ml-lab")).not_to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_clicking_tile_navigates_to_namespace_overview(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    p = NamespacePickerPage(page=page, base_url=live_operator_app.base_url)
    await p.goto()
    await p.tile("aiperf-bench").click()
    # Hash-based router; the URL should end with #/ns/aiperf-bench (no trailing slash).
    await page.wait_for_url(lambda url: url.rstrip("/").endswith("#/ns/aiperf-bench"))
