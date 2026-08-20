from __future__ import annotations

from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"


def _source(*parts: str) -> str:
    return (UI_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def test_router_decodes_route_and_query_segments_safely() -> None:
    source = _source("lib", "router.js")

    assert "function safeDecodeURIComponent(value)" in source
    assert "try" in source and "decodeURIComponent(value)" in source
    assert "if (error instanceof URIError) return value" in source
    assert "out[safeDecodeURIComponent(pair)] = ''" in source
    assert "safeDecodeURIComponent(pair.slice(0, eq))" in source
    assert "safeDecodeURIComponent(pair.slice(eq + 1))" in source
    assert "params[pp.slice(1)] = safeDecodeURIComponent(vp)" in source


def test_launch_rejects_yaml_prototype_pollution_keys() -> None:
    source = _source("pages", "launch.js")

    assert (
        "const DANGEROUS_YAML_KEYS = new Set(['__proto__', 'constructor', 'prototype'])"
        in source
    )
    assert "function assertSafeYamlKey(key, lineNo = null)" in source
    assert (
        "throw new Error(`${location}: key '${key}' is not allowed in launch YAML.`)"
        in source
    )
    assert "Object.getPrototypeOf(value) !== Object.prototype" in source
    assert "assertSafeYamlKey(key)" in source
    assert "return sanitizeParsedYaml(documents[0]);" in source


def test_sweep_detail_uses_archived_child_manifest_fallback() -> None:
    helper_source = _source("pages", "sweep-detail-helpers.js")
    page_source = _source("pages", "sweep-detail.js")

    assert (
        "export function resolveSweepManifest({ detail, archivedChildren })"
        in helper_source
    )
    assert "if (Array.isArray(raw) && raw.length > 0) return raw" in helper_source
    assert (
        "if (raw && Array.isArray(raw.children) && raw.children.length > 0)"
        in helper_source
    )
    assert (
        "if (Array.isArray(archivedChildren) && archivedChildren.length > 0)"
        in helper_source
    )
    assert "return archivedChildren" in helper_source
    assert "api.getSweepChildren(namespace, name, epoch)" in page_source
    assert "setArchivedChildren(d?.children ?? [])" in page_source
    assert "resolveSweepManifest({ detail, archivedChildren })" in page_source


def test_archived_sweep_trials_contribute_to_variation_metrics() -> None:
    source = _source("pages", "sweep-detail-helpers.js")

    assert "for (const c of manifest)" in source
    assert (
        "const idx = Number(pick(c, ['variation_index', 'variationIndex']) ?? 0)"
        in source
    )
    assert "group.n_total += 1" in source
    assert "const summary = childSummaries?.[c.name]?.summary ?? null" in source
    assert "if (summary) group.summaries.push(summary)" in source
    assert "const values = group.summaries" in source
    assert "perMetric[metric.key + '.' + metric.stat] = meanStd(values)" in source
    assert "n_trials: group.summaries.length" in source


def test_sweep_artifact_file_urls_encode_each_path_segment() -> None:
    source = _source("lib", "api.js")

    assert "sweepArtifactFileUrl(ns, sweepName, epoch, filename)" in source
    assert "const nsSeg = encodeURIComponent(ns)" in source
    assert "const sweepSeg = encodeURIComponent(sweepName)" in source
    assert "const epSeg = encodeURIComponent(epoch)" in source
    assert (
        "const fileSeg = filename.split('/').map(encodeURIComponent).join('/')"
        in source
    )
    assert "artifacts/${fileSeg}" in source


def test_log_strip_links_use_semantic_job_routes() -> None:
    source = _source("components", "log-strip.js")

    assert "import { navigate, buildJobPath } from '../lib/router.js'" in source
    assert "const jobPath = buildJobPath({ namespace: e.ns, name: e.name })" in source
    assert "href=${jobPath}" in source
    assert "navigate(jobPath)" in source
    assert "#/jobs/${e.ns}" not in source


def test_compare_uses_namespace_qualified_job_identity() -> None:
    source = _source("pages", "compare.js")

    assert 'selectedKeys are composite "<namespace>/<job_id>" strings' in source
    assert "function compositeKey(job)" in source
    assert "return ns ? `${ns}/${id}` : id" in source
    assert "function splitKey(key)" in source
    assert "return idx < 0 ? { ns: '', jobId: key }" in source
    assert "api.compareJobs(selectedKeys)" in source
    assert "const displayKeys = selectedKeys" in source
    assert "e.values?.[key]" in source


def test_diagnostics_archived_views_fall_back_to_surviving_tabs() -> None:
    panel_source = _source("components", "diagnostics-panel.js")
    job_source = _source("pages", "job-detail.js")

    assert (
        "const availableTabs = useMemo(() => archived ? ['events', 'conditions'] : ALL_TABS, [archived])"
        in panel_source
    )
    assert (
        "const defaultTab = (mode === 'live' && !archived) ? 'events' : 'conditions'"
        in panel_source
    )
    assert "if (!availableTabs.includes(active))" in panel_source
    assert (
        "const renderedActive = availableTabs.includes(active) ? active : defaultTab"
        in panel_source
    )
    assert (
        "mode=${viewingCurrentRun ? (isRunning ? 'live' : 'completed') : 'archived'}"
        in job_source
    )
    assert "archived=${!viewingCurrentRun}" in job_source


def test_relaunch_config_prefill_redacts_raw_secret_values_when_available() -> None:
    source = _source("components", "relaunch-button.js")

    assert "const SENSITIVE_CONFIG_KEYS = [" in source
    for key in [
        "api_key",
        "apiKey",
        "authorization",
        "bearerToken",
        "client_secret",
        "password",
        "secret",
        "secretRef",
        "token",
    ]:
        assert f"'{key}'" in source
    assert "function isSensitiveConfigKey(key)" in source
    assert "export function redactConfigForYaml(value)" in source
    assert (
        "isSensitiveConfigKey(key) ? '[REDACTED]' : redactConfigForYaml(item)" in source
    )
    assert "spec: redactConfigForYaml(spec)" in source
    assert "spec:" in source and "spec: spec" not in source
