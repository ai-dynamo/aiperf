// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Embedded config templates.

pub const TEMPLATES: &[Template] = &[
    Template {
        name: "audio_multimodal",
        title: "Audio/Speech Model Benchmark",
        description: "Benchmark speech-to-text or audio understanding models.",
        category: "Multimodal",
        content: include_str!("../../../../src/aiperf/config/templates/audio_multimodal.yaml"),
    },
    Template {
        name: "dynosim_offline_replay",
        title: "DynoSim Offline Trace Replay",
        description: "Deterministic in-process Dynamo co-simulation of a trace — no server, no sockets.",
        category: "Advanced",
        content: include_str!(
            "../../../../src/aiperf/config/templates/dynosim_offline_replay.yaml"
        ),
    },
    Template {
        name: "embeddings",
        title: "Embeddings Endpoint Benchmark",
        description: "Benchmark embedding models with batched text requests.",
        category: "Specialized Endpoints",
        content: include_str!("../../../../src/aiperf/config/templates/embeddings.yaml"),
    },
    Template {
        name: "env_var_production",
        title: "Environment Variable Production Config",
        description: "CI/CD-friendly template where all deployment-specific values come from env vars.",
        category: "Advanced",
        content: include_str!("../../../../src/aiperf/config/templates/env_var_production.yaml"),
    },
    Template {
        name: "fixed_schedule",
        title: "Fixed Schedule (Hand-Authored Timestamps)",
        description: "Send requests at exact millisecond timestamps from a JSONL file for deterministic temporal testing.",
        category: "Load Testing",
        content: include_str!("../../../../src/aiperf/config/templates/fixed_schedule.yaml"),
    },
    Template {
        name: "goodput_slo",
        title: "Goodput / SLO Benchmark",
        description: "Measure good requests/sec that meet latency SLO thresholds at multiple load levels.",
        category: "Load Testing",
        content: include_str!("../../../../src/aiperf/config/templates/goodput_slo.yaml"),
    },
    Template {
        name: "gpu_telemetry",
        title: "GPU Telemetry (DCGM / pynvml)",
        description: "Collect GPU power, utilization, memory, and temperature during a benchmark via DCGM or pynvml.",
        category: "Advanced",
        content: include_str!("../../../../src/aiperf/config/templates/gpu_telemetry.yaml"),
    },
    Template {
        name: "http_trace_metrics",
        title: "HTTP Trace Metrics (Latency Breakdown)",
        description: "Capture per-phase HTTP timing (DNS, connect, sending, waiting/TTFB, receiving) for transport-layer debugging.",
        category: "Advanced",
        content: include_str!("../../../../src/aiperf/config/templates/http_trace_metrics.yaml"),
    },
    Template {
        name: "inline_dataset",
        title: "Inline Dataset",
        description: "Embed dataset records directly in the YAML config (no separate JSONL file required).",
        category: "Datasets",
        content: include_str!("../../../../src/aiperf/config/templates/inline_dataset.yaml"),
    },
    Template {
        name: "jinja2_variables",
        title: "Jinja2 Computed Config",
        description: "Define variables once and compute derived values with Jinja2 expressions.",
        category: "Advanced",
        content: include_str!("../../../../src/aiperf/config/templates/jinja2_variables.yaml"),
    },
    Template {
        name: "kv_cache_test",
        title: "KV Cache / Prefix Caching",
        description: "Test KV cache efficiency with shared system prompts using user-centric mode.",
        category: "Advanced",
        content: include_str!("../../../../src/aiperf/config/templates/kv_cache_test.yaml"),
    },
    Template {
        name: "latency_test",
        title: "Latency Test (Controlled QPS)",
        description: "Measure TTFT, ITL, and E2E latency at a controlled request rate.",
        category: "Load Testing",
        content: include_str!("../../../../src/aiperf/config/templates/latency_test.yaml"),
    },
    Template {
        name: "long_context",
        title: "Long Context Benchmark (32K+)",
        description: "Test performance with long input contexts and prefill concurrency limits.",
        category: "Advanced",
        content: include_str!("../../../../src/aiperf/config/templates/long_context.yaml"),
    },
    Template {
        name: "minimal",
        title: "Minimal Configuration",
        description: "Bare minimum config using shorthand forms -- the fastest way to get started.",
        category: "Getting Started",
        content: include_str!("../../../../src/aiperf/config/templates/minimal.yaml"),
    },
    Template {
        name: "multi_turn_conversation",
        title: "Multi-Turn Conversation",
        description: "Simulate realistic chatbot workloads with multi-turn context accumulation.",
        category: "Datasets",
        content: include_str!(
            "../../../../src/aiperf/config/templates/multi_turn_conversation.yaml"
        ),
    },
    Template {
        name: "multi_url_load_balancing",
        title: "Multi-URL Load Balancing",
        description: "Distribute requests across multiple server replicas for load balancer testing.",
        category: "Load Testing",
        content: include_str!(
            "../../../../src/aiperf/config/templates/multi_url_load_balancing.yaml"
        ),
    },
    Template {
        name: "multimodal_vision",
        title: "Vision-Language Model Benchmark",
        description: "Benchmark VLMs with synthetic images of varying resolutions.",
        category: "Multimodal",
        content: include_str!("../../../../src/aiperf/config/templates/multimodal_vision.yaml"),
    },
    Template {
        name: "public_dataset",
        title: "Public Dataset (ShareGPT)",
        description: "Use real multi-turn conversations from the ShareGPT public dataset.",
        category: "Datasets",
        content: include_str!("../../../../src/aiperf/config/templates/public_dataset.yaml"),
    },
    Template {
        name: "ramping",
        title: "Gradual Load Ramping",
        description: "Smoothly ramp concurrency and request rate to avoid cold-start transients and connection storms.",
        category: "Load Testing",
        content: include_str!("../../../../src/aiperf/config/templates/ramping.yaml"),
    },
    Template {
        name: "request_cancellation",
        title: "Request Cancellation Test",
        description: "Test server behavior when clients cancel in-flight requests.",
        category: "Load Testing",
        content: include_str!("../../../../src/aiperf/config/templates/request_cancellation.yaml"),
    },
    Template {
        name: "scenario_workload_profiles",
        title: "Scenario Sweep: Workload Profiles",
        description: "Hand-curated named scenarios testing distinct workload shapes.",
        category: "Sweep & Multi-Run",
        content: include_str!(
            "../../../../src/aiperf/config/templates/scenario_workload_profiles.yaml"
        ),
    },
    Template {
        name: "speed_bench_sweep",
        title: "SPEED-Bench Per-Category Sweep",
        description: "Sweep all 11 qualitative SPEED-Bench categories to populate the speed-bench-report matrix.",
        category: "Sweep & Multi-Run",
        content: include_str!("../../../../src/aiperf/config/templates/speed_bench_sweep.yaml"),
    },
    Template {
        name: "sweep_distributions",
        title: "Grid Sweep + Multi-Run",
        description: "Cartesian product sweep over ISL x rate with statistical multi-run aggregation.",
        category: "Sweep & Multi-Run",
        content: include_str!("../../../../src/aiperf/config/templates/sweep_distributions.yaml"),
    },
    Template {
        name: "sweep_with_plot",
        title: "Concurrency Sweep with Inline Plot Envelope",
        description: "Sweep concurrency to draw a Pareto frontier, with the visualization config inlined in the same YAML.",
        category: "Sweep & Multi-Run",
        content: include_str!("../../../../src/aiperf/config/templates/sweep_with_plot.yaml"),
    },
    Template {
        name: "time_based_soak",
        title: "Time-Based Soak Test",
        description: "Run a fixed-duration benchmark (e.g., 1h) to validate stability and detect leaks under sustained load.",
        category: "Load Testing",
        content: include_str!("../../../../src/aiperf/config/templates/time_based_soak.yaml"),
    },
    Template {
        name: "trace_replay",
        title: "Production Trace Replay",
        description: "Replay production traffic from a trace file with exact request timestamps.",
        category: "Datasets",
        content: include_str!("../../../../src/aiperf/config/templates/trace_replay.yaml"),
    },
    Template {
        name: "user_files",
        title: "User Files (Templated Artifacts)",
        description: "Capture run parameters as JSON/YAML/text files materialized into the artifact directory.",
        category: "Advanced",
        content: include_str!("../../../../src/aiperf/config/templates/user_files.yaml"),
    },
    Template {
        name: "warmup_profiling",
        title: "Warmup + Profiling (Two-Phase)",
        description: "Proper benchmark setup: warmup phase for JIT/cache, then clean profiling.",
        category: "Getting Started",
        content: include_str!("../../../../src/aiperf/config/templates/warmup_profiling.yaml"),
    },
];

/// One embedded template.
pub struct Template {
    /// Template identifier.
    pub name: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub category: &'static str,
    /// Raw YAML content, including its SPDX header.
    pub content: &'static str,
}
