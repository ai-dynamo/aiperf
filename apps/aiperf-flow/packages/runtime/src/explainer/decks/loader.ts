/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deck loader for bundled explainer decks.
//!
//! Provides compiled Flow IR for built-in explainer decks. Each deck is
//! pre-compiled and includes metadata for registration in the ExplainerRegistry.

import type { ExplainerDefinition, SlideDefinition } from '@aiperf/flow-compiler';

/**
 * AIPerf Flow System explainer deck definition.
 * Covers request lifecycle, clock-aware execution, measurement, and visualization.
 */
export const AIPERF_FLOW_SYSTEM_DECK: ExplainerDefinition = {
  id: 'aiperf-flow-system',
  route: '/explainers/aiperf-flow-system',
  topic: 'aiperf-architecture',
  eyebrowLabel: 'AIPerf Flow System',
  startGateTitle: 'Explore Request Lifecycle',
  slides: [
    {
      eyebrow: 'Module 1',
      title: 'AIPerf Flow System Architecture',
      lede: 'A clock-aware load generator with integrated visualization and measurement',
      narration:
        'AIPerf Flow is a native Rust load generator and measurement system for inference servers. ' +
        'It combines protocol-aware request dispatch, clock-driven scheduling, and real-time measurement ' +
        'into a single integrated runtime. The Flow visualization system maps this execution model into ' +
        'interactive diagrams that teach the request lifecycle and inference topology.',
      points: [
        'Native Rust CLI for load generation and profiling',
        'Clock-aware scheduling with deterministic simulation support',
        'Protocol-neutral transport abstraction over HTTP and gRPC',
        'Integrated visualization of request journeys and system topology',
      ],
      caption: 'AIPerf unifies load generation, measurement, and visualization of inference systems',
    } as SlideDefinition,

    {
      eyebrow: 'Module 2',
      title: 'Request Journey: Seven Distinct Boundaries',
      lede: 'A single request crosses admission, transport, model, stream, and observation boundaries',
      narration:
        'Every request passes through seven distinct lifecycle boundaries: arrival at the scheduler, ' +
        'admission through a clock-aware queue, dispatch to the transport layer, service beginning in the model, ' +
        'first token emission, terminal stream event, and final observer record. Each boundary represents a causal ' +
        'and temporal milestone that the measurement system captures with stable evidence identifiers.',
      points: [
        'Arrival: scheduler receives the request at wall or virtual time',
        'Admission: clock-aware queue decides to begin service',
        'Dispatch: transport serializes and sends the request',
        'First token: response stream emits its first output token',
        'Terminal: stream closes without error or signals completion',
        'Record: observer finalizes measurement and persists the evidence',
      ],
      caption: 'Request lifecycle evidence flows from arrival through observer finalization',
      term: { word: 'Evidence', meaning: 'A stable, unique identifier for a lifecycle boundary with timestamp and causal context' },
    } as SlideDefinition,

    {
      eyebrow: 'Module 3',
      title: 'Admission Queue: Deterministic Scheduling',
      lede: 'A clock-driven admission boundary that enforces request-rate and concurrency policies',
      narration:
        'The admission queue is the first clock-aware component in the execution pipeline. It receives requests ' +
        'from the client scheduler, applies admission policy based on the configured workload, and dispatches ' +
        'admitted requests to the transport layer at the correct clock time. The queue is policy-agnostic: it can ' +
        'enforce fixed request rates, target concurrency levels, or multi-turn user-centric conversations. Arrival ' +
        'and admission timestamps are captured as stable evidence.',
      points: [
        'Receives requests from the client scheduler',
        'Enforces request-rate, concurrency, or user-centric policies',
        'Dispatches admitted requests to the transport at scheduled time',
        'Preserves arrival and admission timestamps as evidence',
        'Supports both real-wall-time and deterministic simulation modes',
      ],
      caption: 'Clock-aware admission enforces scheduling policy while preserving causality',
      term: { word: 'Clock', meaning: 'Abstraction providing wall time (RealClock) or deterministic virtual time (SimClock) to every scheduled action' },
    } as SlideDefinition,

    {
      eyebrow: 'Module 4',
      title: 'Protocol-Neutral Transport Layer',
      lede: 'HTTP and gRPC endpoints bound through pluggable transport abstractions',
      narration:
        'The transport layer dispatches admitted requests to inference endpoints through protocol-specific ' +
        'implementations. HTTP transport supports HTTP/1, h2c, UDS, TLS, and SSE streaming. gRPC transport ' +
        'supports KServe OIP and NVIDIA Riva endpoint families. Connection establishment includes clock-driven ' +
        'retry logic with configurable backoff. The transport is transport-neutral to the dispatcher: both HTTP ' +
        'and gRPC implement a common sink interface that receives requests and reports response events.',
      points: [
        'HTTP: HTTP/1, h2c, UDS, TLS, Server-Sent Events',
        'gRPC: KServe OIP, NVIDIA Riva ASR, TTS, NLP families',
        'Connection pooling with clock-driven linear backoff retry',
        'Streaming response handling with token-by-token observation',
        'Request-local and worker-local state isolation',
      ],
      caption: 'Transport abstractions decouple protocol details from request lifecycle',
      term: { word: 'Sink', meaning: 'A transport-specific handler that receives one request and drives it to terminal, observing lifecycle events' },
    } as SlideDefinition,

    {
      eyebrow: 'Module 5',
      title: 'Request Sink: Worker-Local Scheduling',
      lede: 'Thread-per-core execution model with request-local state isolation',
      narration:
        'The execution model is thread-per-core: a single worker owns its own scheduling, admission, transport, ' +
        'capture, and measurement state. This design eliminates per-request allocation overhead and contention in ' +
        'the critical measurement path. One-worker deployments use Tokio\'s LocalSet on the coordinator\'s ' +
        'current-thread runtime. Multi-worker runs spawn each worker on its own OS thread, each with a ' +
        'self-contained runtime. A request sink is the worker-local handler that executes one request: it serializes ' +
        'the request, sends it to the endpoint, observes the response stream, and participates in measurement.',
      points: [
        'Thread-per-core execution with local state ownership',
        'Single-worker LocalSet on current-thread Tokio runtime',
        'Multi-worker OS threads with independent scheduling',
        'Worker-local sink owns transport, capture, and measurement',
        'No Arc<Mutex<_>> contention on request or token paths',
      ],
      caption: 'Worker-local architecture eliminates measurement overhead from allocation and contention',
      term: { word: 'Worker', meaning: 'An OS thread or LocalSet that owns a request sink and independent scheduling state' },
    } as SlideDefinition,

    {
      eyebrow: 'Module 6',
      title: 'Observer: Token-by-Token Measurement',
      lede: 'Streaming response observation with first-token and terminal event separation',
      narration:
        'The observer receives events from the response stream and records them as stable evidence. First-token ' +
        'emission is distinct from stream terminal: a response that produces zero tokens has a terminal event but ' +
        'no first-token event. Token arrivals are observed but not recorded individually; instead, the observer ' +
        'accumulates generated-token count and measures inter-token latency from first to terminal. When the stream ' +
        'closes, the observer completes measurement, finalizes usage and accuracy data, and emits the final record ' +
        'to persistent storage. Measurement data includes input and output token counts, first-token latency, ' +
        'generated-token inter-token latency, and endpoint-specific usage observations.',
      points: [
        'First-token and terminal events are observed as separate boundaries',
        'Token stream is preserved as bytes until complete lines available',
        'Generated-token latency measured from first to terminal event',
        'Token counts captured from endpoint usage or local tokenizer',
        'Accuracy and adaptive control observe inference output',
        'Final record emitted after terminal event and observer finalization',
      ],
      caption: 'Token-by-token observation preserves first-token, streaming, and terminal semantics',
      term: { word: 'TTFT', meaning: 'Time-To-First-Token: latency from request dispatch to first output token received' },
    } as SlideDefinition,

    {
      eyebrow: 'Module 7',
      title: 'Evidence-Backed Measurement',
      lede: 'Record, aggregate, and phase-window metrics derived from stable evidence',
      narration:
        'The measurement system operates at four levels: per-record measurement captures every request with its ' +
        'evidence timestamps and inference outputs; aggregation computes summary statistics over a batch of records; ' +
        'phase metrics apply window-based constraints from the workload lifecycle like warmup, ramp, steady-state, ' +
        'and drain; sweep metrics compute derived histograms, percentiles, and rates over configured dimensions. ' +
        'Exact mode retains all records; sketch mode uses mergeable t-digests for throughput and percentile estimates. ' +
        'Steady-state mode derives a measurement window from the in-flight concurrency curve. Every metric is grounded ' +
        'in the stable evidence timestamps, making them reproducible across replays and deterministic simulation.',
      points: [
        'Per-record measurement with evidence-backed timestamps',
        'Exact and sketch-mode aggregation',
        'Phase-window metrics: warmup, ramp, steady-state, drain',
        'Sweep metrics over configured dimensions',
        'Steady-state window derived from concurrency curve',
        'Reproducible across real and simulated runs',
      ],
      caption: 'Evidence-backed measurement ensures reproducibility and correctness',
      term: { word: 'Steady-state', meaning: 'A measurement window derived from the concurrency ramp curve, excluding warmup and drain' },
    } as SlideDefinition,

    {
      eyebrow: 'Module 8',
      title: 'Interactive Request Flow Diagrams',
      lede: 'Real-time visualization of request topology and lifecycle causality',
      narration:
        'The Flow visualization system renders the request lifecycle and system topology as interactive diagrams. ' +
        'A Flow document defines scenes that arrange stable entities like queues, transports, models, and streams ' +
        'spatially, then connects them with relations showing how requests flow. A scene narrates the spatial ' +
        'arrangement, then a timeline orchestrates camera movements, element reveals, and connection traces that ' +
        'teach the lifecycle progression. Interactive selections let viewers inspect entity details, evidence metadata, ' +
        'and causal relations. Responsive layouts adapt to viewport size. Fallback modes preserve narration, reading ' +
        'order, and semantic meaning for SVG and HTML-only contexts.',
      points: [
        'Scenes define spatial topology with stable entities and relations',
        'Timelines choreograph camera, reveals, traces, and narration',
        'Interactive inspection of entities and evidence',
        'Reading order, keyboard navigation, accessibility roles preserved',
        'Reduced-motion support and semantic fallback for non-interactive contexts',
        'Evidence-backed narration tied to lifecycle events',
      ],
      caption: 'Flow visualization teaches request lifecycle through interactive spatial diagrams',
      term: { word: 'Scene', meaning: 'A spatial arrangement of entities and relations with timeline choreography and interaction handlers' },
    } as SlideDefinition,

    {
      eyebrow: 'Module 9',
      title: 'Unified Measurement and Visualization',
      lede: 'AIPerf execution model instrumented through Flow visualization',
      narration:
        'AIPerf Flow integrates load generation, measurement, and visualization into a single system. The AIPerf ' +
        'runtime captures evidence-backed measurement from the request lifecycle. The Flow compiler transforms .flow ' +
        'source into interactive scenes. At runtime, the explainer deck integrates the two: each slide corresponds ' +
        'to a phase or concept in the request lifecycle, with optional embedded scenes that visualize the current ' +
        'topic. Narration is coupled to timeline progression, allowing viewers to see request flow animations while ' +
        'hearing description of the causality and measurement semantics. The integration supports both browser-driven ' +
        'interactive learning and deterministic offline simulation and replay through Dynamo.',
      points: [
        'AIPerf captures stable evidence from request lifecycle',
        'Flow compiler produces interactive visual scenes',
        'Explainer deck couples narration to scene progression',
        'Optional embedded scenes on relevant slides',
        'Support for real and deterministic simulation modes',
        'Keyboard and voice navigation for accessibility',
      ],
      caption: 'AIPerf Flow unifies execution measurement, visualization, and interactive learning',
      term: { word: 'Explainer Deck', meaning: 'A slideshow combining narration, concepts, and optional embedded scenes to teach system architecture' },
    } as SlideDefinition,
  ],
  scenesById: new Map(),
};

/**
 * Registry of all built-in explainer decks.
 * Add new decks here to make them available at runtime.
 */
export const BUILTIN_DECKS = [AIPERF_FLOW_SYSTEM_DECK] as const;

/**
 * Load all built-in decks into the provided registry.
 * Each deck is pre-compiled with full metadata and narration.
 */
export function loadBuiltinDecks(registry: { register: (deck: ExplainerDefinition) => void }): void {
  for (const deck of BUILTIN_DECKS) {
    registry.register(deck);
  }
}
