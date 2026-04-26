# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial: HTML / XSS escaping across the operator UI.

Preact + htm is expected to escape all interpolated strings by default
— but "by default" is a behavioural claim, not an invariant. A single
``dangerouslySetInnerHTML`` or a hand-rolled ``innerHTML =`` anywhere
in the UI tree would leak raw markup into the DOM. These tests pin the
contract at every surface where attacker-controlled text flows in:

- Archive / Namespace overview row: job name cell.
- Run view: condition ``message`` and pod ``metadata.name``.

The contract: (1) no attacker-injected HTML element is created from
the payload (direct DOM check), and (2) the payload's escaped form
appears in the page source (``page.content()``), proving the string
was interpolated as text rather than omitted by sanitisation.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import ArchivePage, JobDetailPage, NamespaceOverviewPage

pytestmark = [pytest.mark.e2e]

# The payload carries a tag that would execute JS if unescaped. We never
# rely on the JS firing (the browser-error gate would catch it anyway);
# the observable contract is that the tag never renders as an element
# and its escaped form is present in the page source.
_XSS_PAYLOAD = "<img src=x onerror=alert('xss')>"


async def _assert_escaped(page, payload: str = _XSS_PAYLOAD) -> None:
    """Assert that ``payload`` appears as escaped text, not as markup.

    The two independent checks each catch a different class of failure:

    - Zero matches for the payload's DOM element (``img[src='x']``)
      catches the straight XSS case — if the string were rendered
      unescaped, the browser would materialise an ``<img>``.
    - The escaped entity form (``&lt;...&gt;``) in ``page.content()``
      catches silent failure modes where the string is dropped by a
      sanitiser rather than escaped — we want to know when the payload
      ends up visible but safe, vs. swallowed entirely.
    """
    assert await page.locator("img[src='x']").count() == 0, (
        "an <img> element materialised from the XSS payload — HTML escaping bypassed"
    )
    html = await page.content()
    escaped = payload.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    assert escaped in html, (
        f"escaped payload not found in page source — payload may have been "
        f"silently dropped rather than escaped. Looked for: {escaped!r}"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_escapes_html_in_job_name(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """A job name containing HTML is rendered as text on ``/ns/<ns>/archive``.

    Kubernetes names can't actually contain these characters (DNS label
    rules), but the operator's job-id-label path is the only thing that
    enforces that — a badly-written CRD controller or a forged response
    could still put arbitrary text in the API payload. The UI must not
    become the weakest link.
    """
    # jobs_raw[0] is in the ``aiperf-bench`` namespace.
    fake_k8s_client.jobs_raw[0]["metadata"]["name"] = _XSS_PAYLOAD
    archive = ArchivePage(page, live_operator_app.base_url, namespace="aiperf-bench")
    await archive.goto()
    await expect(archive.rows().first).to_be_visible()
    await _assert_escaped(page)


@pytest.mark.asyncio(loop_scope="session")
async def test_namespace_overview_escapes_html_in_job_name(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Same contract as archive, on the per-namespace overview run list."""
    fake_k8s_client.jobs_raw[0]["metadata"]["name"] = _XSS_PAYLOAD
    overview = NamespaceOverviewPage(
        page, live_operator_app.base_url, namespace="aiperf-bench"
    )
    await overview.goto()
    await _assert_escaped(page)


@pytest.mark.asyncio(loop_scope="session")
async def test_namespace_name_with_html_chars_renders_escaped(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Namespace tile on the picker escapes HTML in the namespace name.

    The fixture's namespaces are plain ASCII; to exercise the namespace-name
    XSS path, inject a fake job into the running app's ``jobs`` signal via
    ``page.evaluate``. If the app does not expose its signals on
    ``window.__aiperf_state__`` the test is skipped (fixture augmentation
    is intentionally out of scope here).
    """
    nasty = "ns<script>alert(1)</script>"
    await page.goto(live_operator_app.base_url + "/")
    await page.evaluate(f"""
      (() => {{
        const mod = window.__aiperf_state__;
        if (!mod || !mod.jobs) return;
        mod.jobs.value = [
          ...mod.jobs.value,
          {{ namespace: {nasty!r}, name: 'job1', phase: 'Running', startTime: new Date().toISOString() }},
        ];
      }})()
    """)
    tile = page.locator(f"[data-testid='np-tile-{nasty}']")
    if await tile.count() == 0:
        pytest.skip(
            "App does not expose signals on window; XSS test for ns names "
            "requires fixture augmentation (deferred)."
        )
    inner = await tile.inner_html()
    assert "<script>" not in inner


@pytest.mark.asyncio(loop_scope="session")
async def test_run_view_escapes_html_in_condition_message(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """An HTML payload in ``status.conditions[].message`` renders as text.

    The conditions section directly interpolates the message string from
    the apiserver response; if a failing probe (or a malicious admission
    webhook) writes HTML into the message field, it must not end up as
    markup in the browser.
    """
    for raw in fake_k8s_client.jobs_raw:
        if raw["metadata"]["name"] == "aiperf-llama3-c128":
            raw.setdefault("status", {})["conditions"] = [
                {
                    "type": "Ready",
                    "status": "False",
                    "message": f"probe said: {_XSS_PAYLOAD}",
                }
            ]
            break
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    await expect(page.get_by_test_id("run-conditions")).to_be_visible()
    await _assert_escaped(page)


@pytest.mark.asyncio(loop_scope="session")
async def test_run_view_escapes_html_in_pod_name(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """An HTML payload in a pod's ``metadata.name`` renders as text."""
    evil_pod_name = f"worker-{_XSS_PAYLOAD}-0"
    fake_k8s_client.pods_by_job["live-run"] = [
        {
            "metadata": {
                "name": evil_pod_name,
                "namespace": "aiperf-bench",
                "labels": {"aiperf.nvidia.com/job-id": "live-run"},
            },
            "status": {"phase": "Running", "containerStatuses": []},
            "spec": {"containers": [{"name": "worker"}]},
        }
    ]
    # Give the run view a spec so the config fetch doesn't 404.
    import orjson

    live_dir = live_operator_app.results_dir / "aiperf-bench" / "live-run"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "job_spec.json").write_bytes(orjson.dumps({"benchmark": {}}))

    detail = JobDetailPage(page, live_operator_app.base_url, "aiperf-bench", "live-run")
    await detail.goto()
    await expect(page.get_by_test_id("run-pods")).to_be_visible()
    await _assert_escaped(page)
