---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Roadmap
---

# AIPerf Public Release Roadmap

AIPerf is a flagship benchmarking tool for GenAI inference systems. Over the next quarter, we are focusing on making AIPerf more useful for emerging production workloads: agentic inference, speculative decoding, large-scale serving, production trace replay, and hardware telemetry across a broader accelerator ecosystem.

This roadmap will update based on evolving priorities.

## Roadmap Window: June & Q3 2026

June 1, 2026 through September 30, 2026.

## Agentic Benchmarking

We are extending AIPerf to replay agentic task shapes, including agent planning, tool calls, helper agents, and parallel work, so benchmark traffic looks more like real agent activity than a flat list of unrelated requests.

Agentic benchmarks also need faithful replay. AIPerf will preserve the order, timing, and relationships in agentic coding sessions, start measurement from realistic conversation context, and avoid artifacts from replaying the same traces repeatedly. This makes results easier to compare across systems.

Planned work:

- Support replay of standardized multi-turn, tool-using agent traces.
- Add branching conversation support for tasks that split into parallel helper agents and then join back into the main workflow.
- Preserve conversation shape, turn ordering, timing, parent/child relationships, and join points during replay.
- Add agentic replay modes for long-running, steady-state profiles of multi-turn tasks.
- Add agentic-aware metrics such as multi-turn TTFT, end-to-end task latency, conversation-level latency, and separate views for main-agent and helper-agent work.
- Provide benchmark recipes that lock the important replay rules so results are easier to reproduce and compare.
- Improve documentation and examples for linear agent traces, DAG traces, and sub-agent-heavy workloads.

## Speculative Decoding/MTP Benchmarking

As MoE and native-MTP models such as DeepSeek-V3/R1 become more common, speculative decoding is becoming a practical serving concern rather than a niche optimization. AIPerf has initial SPEED-Bench support, and we are making those workflows more reproducible, transparent, and easier to compare across engines where the engines expose the necessary telemetry.

Planned work:

- Correct SPEED-Bench data handling by using the prepared dataset flow rather than masked or placeholder benchmark rows.
- Document the difference between aggregate/server-side acceptance metrics and per-request SpecDec evaluation.
- Add per-request SpecDec telemetry ingestion, starting with SGLang where per-request metric export already exists.
- Report SpecDec behavior alongside normal latency and throughput, including per-request accepted length, token-weighted accepted length, acceptance-length histograms, and category-level views.
- Work toward an engine-neutral telemetry contract for SGLang, TensorRT-LLM, and vLLM, so AIPerf can provide comparable cross-engine SpecDec reporting as framework support matures.

## Kubernetes Scale

We are making AIPerf better able to exercise large inference deployments in the way operators actually run them: distributed, long-running, and Kubernetes-native.

Planned work:

- Validate a reproducible path toward very high concurrent-connection benchmarks.
- Improve Kubernetes deployment recipes and operator documentation.
- Harden large-scale benchmark execution so users can reproduce results with less manual setup.

## Production Trace Replay

Many teams already have production traffic traces. AIPerf already supports several real-world trace replay paths, including Mooncake, Bailian/Qwen usage traces, BurstGPT, Amazon SageMaker Data Capture, and generic raw-payload replay. We are broadening and hardening that surface so more adopters can bring existing traffic into AIPerf without building one-off converters.

Planned work:

- Add and refine adapters for additional production trace formats.
- Improve trace conversion documentation, examples, and validation.
- Preserve workload structure that affects serving performance, including timing, request shape, literal payloads, and multi-turn behavior where available.
- Continue incorporating broadly useful adopter-driven trace and integration work into the shared replay framework.

## Hardware, Power, and Multi-Vendor Telemetry

AIPerf reports common API-level serving metrics today, and it also collects hardware telemetry where supported. As AIPerf grows its adoption across hardware vendors, hardware metrics need clear names, clear ownership, and clear reporting boundaries.

Planned work:

- Publish a namespace convention for vendor-specific hardware telemetry.
- Separate common serving metrics from hardware-specific counters in reports and UI surfaces.
- Make power, energy, and system telemetry easier to collect alongside benchmark runs where the underlying platform exposes it.
- Support a plugin-oriented path for accelerator-specific telemetry providers, starting from existing NVIDIA and ROCm-style collection paths.
