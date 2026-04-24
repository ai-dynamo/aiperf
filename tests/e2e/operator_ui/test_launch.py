# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the Launch view (``/launch``).

Covers the YAML editor + template picker + POST-to-create flow in
``src/aiperf/operator/ui/views/launch.js``. The view parses the editor's
YAML client-side, surfaces parse errors inline (``launch-parse-err``),
and on valid YAML posts to ``/api/v1/jobs`` (the backing
``_create_job_impl``). On success, a ``launch-success`` banner appears
with a ``launch-view-run`` button that deep-links to the new run's
workbench page; on HTTP failure a ``launch-err`` banner surfaces the
server's detail message.

Submissions are recorded on ``FakeK8sClient.created_jobs`` via the
``_create_job_stub`` monkey-patch in ``conftest.py``; tests assert on
that recorder rather than against a real Kubernetes cluster.
"""

from __future__ import annotations

import re

import pytest
from playwright.async_api import expect

from ._pages import LaunchPage

pytestmark = [pytest.mark.e2e]


# A minimal CR manifest that passes the backend's `metadata.name` check.
# The fake ``_create_job_stub`` only inspects ``metadata.name`` /
# ``metadata.namespace``; the rest of ``spec`` is opaque to it.
_MINIMAL_YAML = """
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: e2e-launch-smoke
  namespace: e2e-bench
spec:
  model: llama3-8b
""".strip()


@pytest.mark.asyncio(loop_scope="session")
async def test_launch_page_loads(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """``/launch`` renders the page root + the three template buttons."""
    launch = LaunchPage(page, live_operator_app.base_url)
    await launch.goto()
    for tid in ("llama3-70b-throughput", "mistral-burst", "minimal"):
        await expect(page.get_by_test_id(f"launch-template-{tid}")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_launch_picking_template_populates_yaml(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking a template button fills the YAML textarea.

    ``pickTemplate`` in ``launch.js`` writes the template's ``yaml`` field
    into ``setYaml``, which binds to the ``launch-yaml`` textarea value.
    """
    launch = LaunchPage(page, live_operator_app.base_url)
    await launch.goto()
    textarea = page.get_by_test_id("launch-yaml")
    await launch.pick_template("minimal")
    # The "minimal" template's YAML body always carries the apiVersion header.
    await expect(textarea).to_have_value(re.compile("aiperf.nvidia.com"))


@pytest.mark.asyncio(loop_scope="session")
async def test_launch_submit_records_manifest_and_shows_success(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Submitting a valid manifest records it and shows the success banner.

    The stub in ``conftest.py`` appends ``(namespace, name, manifest)`` to
    ``fake_k8s_client.created_jobs`` and returns a ``CreateJobResponse``;
    the UI reacts by rendering the ``launch-success`` banner with a
    ``launch-view-run`` button.
    """
    launch = LaunchPage(page, live_operator_app.base_url)
    await launch.goto()
    await launch.set_yaml(_MINIMAL_YAML)
    await launch.submit()

    # Success banner appears synchronously after the POST resolves.
    success = page.get_by_test_id("launch-success")
    await expect(success).to_be_visible()
    await expect(success).to_contain_text("e2e-bench")
    await expect(success).to_contain_text("e2e-launch-smoke")
    await expect(page.get_by_test_id("launch-view-run")).to_be_visible()

    # The recorder captured the submission under the manifest's ns/name.
    entries = [(ns, name) for (ns, name, _manifest) in fake_k8s_client.created_jobs]
    assert ("e2e-bench", "e2e-launch-smoke") in entries, entries


@pytest.mark.asyncio(loop_scope="session")
async def test_launch_view_run_deep_links_to_new_run(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Clicking ``launch-view-run`` navigates to the new run's workbench.

    After success, ``viewRun`` in ``launch.js`` calls
    ``navigate(/run/<ns>/<name>)``. The target page mounts even though
    the CR doesn't actually exist in the fake — ``_get_job_impl`` falls
    back to the archived branch when it can't find a CR or PVC dir, but
    for a fresh non-existent run it simply 404s. The URL change itself
    is the observable contract we pin here.
    """
    launch = LaunchPage(page, live_operator_app.base_url)
    await launch.goto()
    await launch.set_yaml(_MINIMAL_YAML)
    await launch.submit()
    await expect(page.get_by_test_id("launch-success")).to_be_visible()
    await page.get_by_test_id("launch-view-run").click()
    await page.wait_for_url("**/run/e2e-bench/e2e-launch-smoke")


@pytest.mark.asyncio(loop_scope="session")
async def test_launch_parse_error_surfaces_inline(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Invalid YAML surfaces ``launch-parse-err`` without firing a POST.

    ``peekManifest`` in ``launch.js`` parses on every keystroke and
    writes the error message into the ``launch-parse-err`` banner.
    ``canSubmit`` becomes false while the parse error is present, so
    clicking submit is also expected to be a no-op.
    """
    launch = LaunchPage(page, live_operator_app.base_url)
    await launch.goto()
    # A bare scalar line is neither ``key: value`` nor ``- item``; the
    # custom YAML parser in ``launch.js`` throws with an explicit line
    # number in that case (see ``parseYaml`` L226).
    await launch.set_yaml("metadata:\n  name: ok\nthis-is-not-valid-yaml")
    await expect(page.get_by_test_id("launch-parse-err")).to_be_visible()
    # Submit button should be disabled while the parse error is live.
    await expect(page.get_by_test_id("launch-submit")).to_be_disabled()
    assert fake_k8s_client.created_jobs == []
