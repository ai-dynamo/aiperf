# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page-object wrappers used by the e2e UI tests.

Thin helpers — not a full POM. Each page exposes ``.goto(...)`` and the
handful of interactions the per-page test files actually exercise.

The operator UI was rewritten to the WORKBENCH shell (see
``src/aiperf/operator/ui/app.js``); routes and test-ids changed
substantially from the prior Flight-Deck incarnation. After Task 11
the legacy Home view and unprefixed routes have been retired; this
module targets the current canonical contract:

- ``/``                  → Namespace picker, root ``page-namespace-picker``
- ``/ns/:ns``            → Namespace overview, root ``page-namespace-overview``
- ``/ns/:ns/launch``     → Launch view, root ``page-launch``
- ``/ns/:ns/archive``    → Archive view, root ``page-archive``
- ``/ns/:ns/run/:name``  → Single-run workbench, root ``page-job-detail``
- ``/compare`` / ``/analysis`` → Analysis view, root ``page-leaderboard``
- ``/log``               → Log view, root ``page-history``
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
        # like ``/archive`` return 404. Route all navigations through
        # ``/#<route>``, with ``/`` short-circuited to the bare index.
        suffix = "" if route in ("", "/") else f"#{route}"
        await self.page.goto(self.base_url + "/" + suffix)


@dataclass
class ArchivePage(BasePage):
    """The ``/ns/:ns/archive`` namespace-scoped past-runs browser."""

    namespace: str = "aiperf-bench"

    async def goto(self) -> None:
        await self._goto(f"/ns/{self.namespace}/archive")
        await expect(self.page.get_by_test_id("page-archive")).to_be_visible()

    def row(self, name: str) -> Locator:
        return self.page.get_by_test_id(f"arch-row-{self.namespace}-{name}")

    def search(self) -> Locator:
        return self.page.get_by_test_id("arch-search")

    def rows(self) -> Locator:
        """All rows currently rendered in the namespace-scoped archive."""
        return self.page.locator(f"[data-testid^='arch-row-{self.namespace}-']")

    async def set_sort(self, value: str) -> None:
        """Set the ``arch-sort`` dropdown to one of its option keys."""
        await self.page.get_by_test_id("arch-sort").select_option(value)


class JobDetailPage(BasePage):
    """The ``/ns/:ns/run/:name`` single-run workbench (root ``page-job-detail``)."""

    def __init__(self, page: Page, base_url: str, namespace: str, name: str) -> None:
        super().__init__(page, base_url)
        self.namespace = namespace
        self.name = name

    async def goto(self) -> None:
        await self._goto(f"/ns/{self.namespace}/run/{self.name}")
        await expect(self.page.get_by_test_id("page-job-detail")).to_be_visible()

    async def cancel(self) -> None:
        await self.page.get_by_test_id("run-cancel").click()


# Backwards-compat alias for callers transitioning to the new name.
RunPage = JobDetailPage


class AnalysisPage(BasePage):
    """The ``/compare`` / ``/analysis`` view (root ``page-leaderboard``)."""

    async def goto(self) -> None:
        await self._goto("/compare")
        await expect(self.page.get_by_test_id("page-leaderboard")).to_be_visible()


class LogPage(BasePage):
    """The ``/log`` / ``/history`` durable run-log view (root ``page-history``)."""

    async def goto(self) -> None:
        await self._goto("/log")
        await expect(self.page.get_by_test_id("page-history")).to_be_visible()


@dataclass
class LaunchPage(BasePage):
    """The ``/ns/:ns/launch`` namespace-aware Launch view (root ``page-launch``).

    The launch view auto-fills ``namespace: <ns>`` from the URL, and locks
    the LAUNCH submit button when the YAML's top-level ``namespace:``
    diverges from the URL segment. The breadcrumb namespace pill flips
    to a ``ns-switcher-pill--bad`` class while divergence is active.
    """

    namespace: str = "aiperf-bench"

    async def goto(self) -> None:
        await self._goto(f"/ns/{self.namespace}/launch")
        await expect(self.page.get_by_test_id("page-launch")).to_be_visible()

    def editor(self) -> Locator:
        return self.page.get_by_test_id("launch-editor")

    def submit(self) -> Locator:
        return self.page.get_by_test_id("launch-submit")

    async def pick_template(self, template_id: str) -> None:
        await self.page.get_by_test_id(f"launch-template-{template_id}").click()

    async def set_yaml(self, yaml_text: str) -> None:
        await self.editor().fill(yaml_text)


class NamespacePickerPage(BasePage):
    """The ``/`` view — cross-namespace picker with one tile per namespace."""

    async def goto(self) -> None:
        await self._goto("/")
        await expect(self.page.get_by_test_id("page-namespace-picker")).to_be_visible()

    def tile(self, namespace: str) -> Locator:
        return self.page.get_by_test_id(f"np-tile-{namespace}")

    def search(self) -> Locator:
        return self.page.get_by_test_id("np-search")


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


@dataclass
class NamespaceOverviewPage(BasePage):
    namespace: str = "aiperf-bench"

    async def goto(self) -> None:
        await self._goto(f"/ns/{self.namespace}")
        await expect(
            self.page.get_by_test_id("page-namespace-overview")
        ).to_be_visible()

    def stats(self) -> Locator:
        return self.page.get_by_test_id("no-stats")

    def row(self, name: str) -> Locator:
        return self.page.get_by_test_id(f"no-row-{self.namespace}-{name}")
