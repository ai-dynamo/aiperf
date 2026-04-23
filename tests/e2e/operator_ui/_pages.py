# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page-object wrappers used by the e2e UI tests.

Thin helpers — not a full POM. Each page exposes `.goto(...)` and the
handful of interactions the per-page test files actually exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

from playwright.async_api import Locator, Page, expect


@dataclass
class BasePage:
    page: Page
    base_url: str

    async def _goto(self, route: str) -> None:
        # The UI uses hash-based routing (``window.location.hash``); the
        # FastAPI server mounts only ``/`` for the SPA, so non-root URLs
        # like ``/jobs`` return 404. Route all navigations through
        # ``/#<route>``, with ``/`` short-circuited to the bare index.
        suffix = "" if route in ("", "/") else f"#{route}"
        await self.page.goto(self.base_url + "/" + suffix)


class DashboardPage(BasePage):
    async def goto(self) -> None:
        await self._goto("/")
        await expect(self.page.get_by_test_id("page-dashboard")).to_be_visible()

    def kpi(self, label: str) -> Locator:
        return self.page.get_by_test_id(f"kpi-{label}")


class JobsPage(BasePage):
    async def goto(self) -> None:
        await self._goto("/jobs")
        await expect(self.page.get_by_test_id("page-jobs")).to_be_visible()

    def rows(self) -> Locator:
        return self.page.get_by_test_id("job-table").locator("[data-testid^='job-row-']")

    def row(self, namespace: str, name: str) -> Locator:
        return self.page.get_by_test_id(f"job-row-{namespace}-{name}")

    async def click_column_header(self, key: str) -> None:
        await self.page.get_by_test_id(f"col-header-{key}").click()

    async def set_namespace_filter(self, ns: str) -> None:
        # The jobs page has no dedicated namespace selector; instead the
        # search input matches against both name and namespace. Typing the
        # namespace narrows the table the same way a filter would.
        await self.page.get_by_placeholder("Search name...").fill(ns)


class JobDetailPage(BasePage):
    def __init__(self, page: Page, base_url: str, namespace: str, name: str) -> None:
        super().__init__(page, base_url)
        self.namespace = namespace
        self.name = name

    async def goto(self) -> None:
        await self._goto(f"/jobs/{self.namespace}/{self.name}")
        await expect(self.page.get_by_test_id("page-job-detail")).to_be_visible()

    async def cancel(self) -> None:
        await self.page.get_by_test_id("job-detail-cancel").click()


class LeaderboardPage(BasePage):
    async def goto(self) -> None:
        await self._goto("/leaderboard")
        await expect(self.page.get_by_test_id("page-leaderboard")).to_be_visible()

    async def select_metric(self, metric: str) -> None:
        await self.page.get_by_test_id("metric-selector").select_option(metric)


class ComparePage(BasePage):
    async def goto(self) -> None:
        await self._goto("/compare")
        await expect(self.page.get_by_test_id("page-compare")).to_be_visible()


class HistoryPage(BasePage):
    async def goto(self) -> None:
        await self._goto("/history")
        await expect(self.page.get_by_test_id("page-history")).to_be_visible()


class CommandPalette:
    def __init__(self, page: Page) -> None:
        self.page = page

    async def open(self) -> None:
        await self.page.keyboard.press("Control+k")
        await expect(self.page.get_by_test_id("command-palette")).to_be_visible()

    async def type(self, text: str) -> None:
        await self.page.get_by_test_id("command-palette-input").fill(text)

    async def press_enter(self) -> None:
        await self.page.get_by_test_id("command-palette-input").press("Enter")
