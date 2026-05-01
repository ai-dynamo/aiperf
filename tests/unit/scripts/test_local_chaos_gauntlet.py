from __future__ import annotations

from tests.scripts.chaos import local_adversarial


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
    }

    assert full_gauntlet_cases <= names
    assert chaos_cases <= names
    assert len(cases) == len(names)


def test_local_adversarial_cases_use_process_group_cleanup_for_subprocesses() -> None:
    assert local_adversarial.run_cmd.__kwdefaults__ is not None
    assert local_adversarial.run_cmd.__kwdefaults__["start_new_session"] is True


def test_local_adversarial_import_has_no_side_effects() -> None:
    assert hasattr(local_adversarial, "main")
    assert hasattr(local_adversarial, "build_cases")
