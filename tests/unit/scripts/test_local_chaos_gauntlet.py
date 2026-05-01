from __future__ import annotations

import pytest

from tests.scripts.chaos import harness, local_adversarial


def test_local_adversarial_case_inventory_is_superset_of_historical_gauntlets() -> None:
    cases = local_adversarial.build_cases()
    names = {case.name for case in cases}

    full_gauntlet_cases = {
        "root-help",
        "version",
        "profile-help",
        "config-help",
        "config-init-help",
        "plugins-help",
        "plot-help",
        "synthesize-help",
        "validate-help",
        "analyze-trace-help",
        "speed-bench-report-help",
        "plugins-default",
        "plugins-all",
        "plugins-validate",
        "plugins-category",
        "plugins-specific",
        "config-init-list",
        "config-init-search",
        "config-init-template",
        "config-generate-yaml",
        "config-generate-json",
        "config-generate-gpu-telemetry-tokens",
        "config-validate",
        "config-show",
        "config-schema",
        "config-diff",
        "profile-from-config",
        "profile-template-config",
        "profile-chat",
        "profile-chat-streaming-headers-extra-server-count",
        "profile-chat-ready-transport",
        "profile-chat-ready-mode-interval",
        "profile-chat-sticky-sessions",
        "profile-multi-url",
        "profile-local-api-port",
        "profile-no-server-metrics",
        "profile-completions-streaming-legacy",
        "profile-responses",
        "profile-embeddings",
        "profile-chat-embeddings",
        "profile-nim-embeddings",
        "profile-nim-rankings",
        "profile-hf-tei-rankings",
        "profile-cohere-rankings",
        "profile-huggingface-generate",
        "profile-huggingface-generate-streaming",
        "profile-image-generation",
        "profile-video-generation",
        "profile-video-generation-form-data-audio",
        "profile-image-retrieval",
        "profile-solido-rag",
        "profile-concurrency",
        "profile-request-rate-constant",
        "profile-request-rate-gamma",
        "profile-duration",
        "profile-warmup",
        "profile-warmup-prefill-fallback",
        "profile-cancellation",
        "profile-sweep-repeated",
        "profile-sweep-independent",
        "profile-synthetic-seq-dist",
        "profile-prefix-pool",
        "profile-context-prompts",
        "profile-multimodal-image",
        "profile-multimodal-audio",
        "profile-multimodal-video",
        "profile-output-exports",
        "profile-export-file",
        "profile-server-metrics",
        "profile-gpu-telemetry",
        "profile-gpu-telemetry-dashboard",
        "profile-single-turn-dataset",
        "profile-multi-turn-dataset",
        "profile-mooncake-fixed-schedule",
        "profile-num-conversations",
        "synthesize-agentic-code",
        "validate-mooncake-trace",
        "analyze-trace",
        "plot-fixtures",
        "speed-bench-report-table",
        "speed-bench-report-csv",
    }
    chaos_cases = {
        "artifact-path-is-file",
        "read-only-artifact-parent",
        "duplicate-local-api-port",
        "bad-config-unknown-nested",
        "bad-template-response-field",
        "custom-endpoint-404",
        "server-500",
        "malformed-json",
        "slow-timeout",
        "interrupt-profile",
        "same-artifact-concurrent",
        "success-output-integrity",
        "network-latency-2s",
        "network-bandwidth-cap",
        "network-reset-peer",
        "network-slow-close",
        "network-timeout-toxic",
    }
    fuzz_cases = {
        "fuzz-numeric-args",
        "fuzz-flag-combos",
        "fuzz-config-yaml",
    }
    resource_cases = {
        "resource-cpu-quota",
        "resource-memory-cap",
        "resource-low-fd-limit",
    }

    assert full_gauntlet_cases <= names
    assert chaos_cases <= names
    assert fuzz_cases <= names
    assert resource_cases <= names
    assert len(cases) == len(names)


def test_local_adversarial_cases_use_process_group_cleanup_for_subprocesses() -> None:
    assert harness.run_cmd.__kwdefaults__ is not None
    assert harness.run_cmd.__kwdefaults__["start_new_session"] is True


def test_local_adversarial_import_has_no_side_effects() -> None:
    assert hasattr(local_adversarial, "main")
    assert hasattr(local_adversarial, "build_cases")


def test_network_cases_skip_gracefully_when_toxiproxy_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.scripts.chaos import chaos_cases, toxiproxy_local

    monkeypatch.setattr(toxiproxy_local, "find_toxiproxy_bin", lambda: None)
    monkeypatch.setattr(chaos_cases, "find_toxiproxy_bin", lambda: None)
    cases = chaos_cases.build_network_cases()
    assert {case.name for case in cases} == {
        "network-latency-2s",
        "network-bandwidth-cap",
        "network-reset-peer",
        "network-slow-close",
        "network-timeout-toxic",
    }
    for case in cases:
        assert case.expected == "SKIP_UNSUPPORTED"


def test_resource_cases_skip_when_not_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.scripts.chaos import resource_cases as rc

    monkeypatch.setattr(rc, "is_linux", lambda: False)
    cases = rc.build_resource_cases()
    assert {case.name for case in cases} == {
        "resource-cpu-quota",
        "resource-memory-cap",
        "resource-low-fd-limit",
    }
    for case in cases:
        assert case.expected == "SKIP_UNSUPPORTED"


def test_resource_cases_skip_individual_helpers_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.scripts.chaos import resource_cases as rc

    monkeypatch.setattr(rc, "is_linux", lambda: True)
    monkeypatch.setattr(rc, "has_systemd_run", lambda: False)
    monkeypatch.setattr(rc, "has_prlimit", lambda: False)
    cases = rc.build_resource_cases()
    for case in cases:
        assert case.expected == "SKIP_UNSUPPORTED"


def test_fuzz_runner_flags_crash_via_pass_required(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.scripts.chaos import fuzz_cases as fc
    from tests.scripts.chaos.harness import (
        Context,
        verdict_for,
    )

    monkeypatch.setenv("AIPERF_FUZZ_MAX_EXAMPLES", "2")
    monkeypatch.setenv("AIPERF_FUZZ_SEED", "0xDEAD")

    crash_text = "Traceback (most recent call last):\n  RuntimeError: boom\n"

    def _fake_run_one(cmd, ctx, log, header):
        with log.open("a") as out:
            out.write(crash_text)
        return 1, crash_text

    monkeypatch.setattr(fc, "_run_one", _fake_run_one)

    ctx = Context(
        base=tmp_path,
        url="http://127.0.0.1:0",
        root=tmp_path,
        logs=tmp_path,
        artifacts=tmp_path,
        fixtures=tmp_path,
        env={},
    )
    case = next(c for c in fc.build_fuzz_cases() if c.name == "fuzz-numeric-args")
    rc, text = case.run(ctx, case.name, tmp_path / "fuzz.log")
    assert rc == 1
    assert "FUZZ_SUMMARY: 2/2 examples crashed" in text
    assert verdict_for(case.expected, rc, text) == "BUG_CRASH"


def test_fuzz_runner_passes_when_no_crash(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.scripts.chaos import fuzz_cases as fc
    from tests.scripts.chaos.harness import Context, verdict_for

    monkeypatch.setenv("AIPERF_FUZZ_MAX_EXAMPLES", "3")

    def _fake_run_one(cmd, ctx, log, header):
        with log.open("a") as out:
            out.write(f"$ {' '.join(cmd)}\nrc=2 (graceful failure)\n")
        return 2, "validation error: ok"

    monkeypatch.setattr(fc, "_run_one", _fake_run_one)

    ctx = Context(
        base=tmp_path,
        url="http://127.0.0.1:0",
        root=tmp_path,
        logs=tmp_path,
        artifacts=tmp_path,
        fixtures=tmp_path,
        env={},
    )
    case = next(c for c in fc.build_fuzz_cases() if c.name == "fuzz-flag-combos")
    rc, text = case.run(ctx, case.name, tmp_path / "fuzz.log")
    assert rc == 0
    assert "FUZZ_SUMMARY: 0/3 examples crashed" in text
    assert verdict_for(case.expected, rc, text) == "OK_PASS"
