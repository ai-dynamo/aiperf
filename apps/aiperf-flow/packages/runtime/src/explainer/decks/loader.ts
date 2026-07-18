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
 * SLURM + Velo explainer deck definition.
 * Covers distributed execution, cluster orchestration, cross-host communication, topology, rank mapping, and message flow.
 */
export const SLURM_VELO_DECK: ExplainerDefinition = {
  id: 'slurm-velo',
  route: '/explainers/slurm-velo',
  topic: 'distributed-execution',
  eyebrowLabel: 'Cluster Orchestration',
  startGateTitle: 'Ready to learn SLURM + Velo?',
  slides: [
    {
      eyebrow: 'The problem',
      title: 'You want to load-test a big AI server',
      lede: 'AIPerf sends many requests to an inference server and measures how fast it answers. To push a really large server hard, one computer sending traffic is not enough — you need many computers sending at once.',
      narration: 'Large AI servers need many load generators acting together as one benchmark.',
      term: { word: 'Inference server', meaning: 'The service that runs an AI model and answers requests (for example, a chat completion endpoint).' },
      points: [
        'One laptop can only send so many requests per second.',
        'To stress a large server you need a fleet of machines generating load together.',
        'That fleet has to act like one coordinated test, not many disconnected ones.',
      ],
      caption: 'Goal: many machines, one benchmark, one result.',
    } as SlideDefinition,

    {
      eyebrow: 'The tool',
      title: 'SLURM hands you a cluster of machines',
      lede: 'SLURM is the software that shares a big cluster among many people. You ask it for machines; it finds free ones, reserves them for you, and runs your command on every one of them at the same time.',
      narration: 'SLURM reserves cluster machines and launches your program across all of them.',
      term: { word: 'SLURM', meaning: 'A job scheduler for compute clusters. You submit a request for N machines; SLURM grants an allocation and launches your program on each task.' },
      points: [
        '`sbatch` submits a batch job; `srun` launches tasks right now.',
        'An allocation is your reserved set of machines for this job.',
        'SLURM runs the exact same command on every task in the allocation.',
      ],
      caption: 'SLURM = the landlord that lends you machines for a while.',
    } as SlideDefinition,

    {
      eyebrow: 'The key trick',
      title: 'Every machine runs the identical command',
      lede: 'This is the part that surprises newcomers. SLURM does not run a different program on each machine. It launches the very same line — `aiperf slurm run` — on all of them at once.',
      narration: 'Every task runs the same AIPerf command, then its rank determines its role.',
      term: { word: 'Task', meaning: 'One running copy of your command inside the allocation. A 4-machine job has 4 tasks.' },
      points: [
        'So how does each copy know what to do differently?',
        'SLURM gives each task a numbered identity called its rank.',
        'The program reads that number and decides its own job.',
      ],
      caption: 'Same command everywhere — the rank number breaks the tie.',
    } as SlideDefinition,

    {
      eyebrow: 'Splitting the roles',
      title: 'Rank 0 leads; everyone else does the work',
      lede: 'AIPerf reads the rank SLURM assigned. Rank 0 becomes the controller — the coordinator. Every other rank becomes a cell — a worker that actually sends load to the server.',
      narration: 'Rank zero coordinates the benchmark. Every other rank becomes a load-generating cell.',
      term: { word: 'Rank', meaning: 'Each task\'s index in the allocation, from 0 up. AIPerf uses it to pick controller vs. cell.' },
      points: [
        'Rank 0 → the single controller (it coordinates, it does not send benchmark load).',
        'Ranks 1, 2, 3, … → cells, numbered cell_id = rank − 1.',
        'So a 4-task job = 1 controller + 3 cells.',
      ],
      caption: 'controller = rank 0 · cell_id = rank − 1 · cell_count = tasks − 1.',
    } as SlideDefinition,

    {
      eyebrow: 'Finding each other',
      title: 'Cells dial the controller with one shared fact',
      lede: 'The cells need to talk to the controller, but nobody set up a directory service. Instead, every task computes the same address from the SLURM environment: the first machine in the allocation, on a known port.',
      narration: 'Each cell derives the rank-zero controller address from the shared SLURM allocation.',
      term: { word: 'Coordinate', meaning: 'The controller\'s address, like tcp://node01:9500, derived identically on every task from SLURM_* variables.' },
      points: [
        'SLURM tells every task the list of machines and its own rank.',
        'All tasks agree rank 0 lives on the first machine in that list.',
        'Default connection port is 9500, so every cell knows exactly where to call.',
      ],
      caption: 'One fact, computed the same everywhere — no discovery service needed.',
    } as SlideDefinition,

    {
      eyebrow: 'Meet Velo',
      title: 'Velo is the walkie-talkie between controller and cells',
      lede: 'Once cells know the controller address, they still need a messaging system. That system is Velo: a small control-plane library AIPerf uses so the controller and cells can exchange short messages across machines.',
      narration: 'Velo carries coordination messages between the controller and its remote cells.',
      term: { word: 'Velo', meaning: 'An async messaging framework used only for cellular control traffic — registration, START sync, heartbeats, and result summaries. It is never on the per-request hot path.' },
      points: [
        'Velo is how AIPerf processes talk to each other across hosts.',
        'It carries small control messages, not the benchmark requests themselves.',
        'Without cellular mode (`--cells`), Velo is not constructed at all.',
      ],
      caption: 'SLURM launches the processes. Velo lets those processes talk.',
    } as SlideDefinition,

    {
      eyebrow: 'Velo bootstrap',
      title: 'A cell connects once, then Velo learns the peer',
      lede: 'Each cell calls Velo with the one known address: tcp://HOST:9500. Velo\'s hello handshake discovers the controller\'s real peer identity, then the two sides can send named messages to each other.',
      narration: 'Each cell connects once, allowing Velo to establish the peer relationship.',
      term: { word: 'Handshake', meaning: 'Velo\'s connect step that turns a host:port into a live, mutually registered peer connection.' },
      points: [
        'The cell only needs AIPERF_CELL_CONTROLLER_ADDR.',
        'Velo connects, shakes hands, and registers both peers.',
        'After that, named handlers like register / heartbeat / partition work.',
      ],
      caption: 'Address in → peer connection out. No service discovery backend.',
    } as SlideDefinition,

    {
      eyebrow: 'Getting ready together',
      title: 'Register and START travel over Velo',
      lede: 'Before any traffic flies, each cell registers over Velo. The controller replies with that cell\'s work slice and a START event handle. When every expected cell has registered, the controller triggers START and they begin together.',
      narration: 'Cells register through Velo, then wait until the controller broadcasts START.',
      term: { word: 'START barrier', meaning: 'A Velo event gate that holds every cell until all of them have registered, so the benchmark starts in sync.' },
      points: [
        'aiperf.cell.register — cell joins and receives its sliced envelope.',
        'The controller waits until cell_count cells have registered.',
        'One START trigger releases every waiting cell at once.',
      ],
      caption: 'Line everyone up over Velo, then start the race together.',
    } as SlideDefinition,

    {
      eyebrow: 'Doing the work',
      title: 'Benchmark requests do NOT use Velo',
      lede: 'After START, cells send their share of requests straight to the inference server over HTTP or gRPC. Velo stays out of that path — it only carries lightweight heartbeats back to the controller.',
      narration: 'Benchmark requests travel directly from cells to the inference server, never through Velo.',
      term: { word: 'Hot path', meaning: 'The per-request and per-token traffic that measures latency. In AIPerf that path is HTTP/gRPC, never Velo.' },
      points: [
        'Each cell sends only its assigned slice — no overlap, no gaps.',
        'Request traffic goes cell → inference server directly.',
        'The controller is not a bottleneck: it never sits in the request path.',
      ],
      caption: 'Velo coordinates. HTTP/gRPC generates the measured load.',
    } as SlideDefinition,

    {
      eyebrow: 'Three planes',
      title: 'Three completely different kinds of traffic',
      lede: 'It helps to keep three roads separate in your head. Velo is only the control road. The load road hits the AI server. Large result files take a third bulk-upload road.',
      narration: 'Control, benchmark traffic, and bulk artifacts use three deliberately separate paths.',
      term: { word: 'Traffic plane', meaning: 'A separate purpose for network traffic: control messages, benchmark load, or bulk artifact files.' },
      points: [
        'Velo — register, START, heartbeats, result partitions / stores.',
        'HTTP / gRPC — the real benchmark requests to the inference server.',
        'HTTP/1 + zstd — large per-record artifact files, not carried on Velo.',
      ],
      caption: 'Mixing these up is the main source of confusion.',
    } as SlideDefinition,

    {
      eyebrow: 'One answer',
      title: 'Result partitions return to rank 0 over Velo',
      lede: 'When the run ends, each cell ships its measurements back to the original rank-0 controller over Velo. The controller merges every cell\'s numbers into one report.',
      narration: 'When work finishes, each cell returns its result partition to rank zero over Velo.',
      term: { word: 'Partition', meaning: 'One cell\'s bundle of results (raw records or a pre-summarized store) sent to the controller at the end over Velo.' },
      points: [
        'aiperf.cell.partition or store_partition — one terminal ship per cell.',
        'Merge happens inside the original rank-0 controller process.',
        'The report looks like one benchmark, not N separate jobs.',
      ],
      caption: 'Cells measure. Rank 0 merges. One authoritative report.',
    } as SlideDefinition,

    {
      eyebrow: 'Bulk files',
      title: 'Huge per-record files take a different road',
      lede: 'If the run keeps large per-record artifact files, those bytes do not ride Velo. They upload over a separate HTTP/1 path with zstd compression. The controller concatenates them after every cell finishes.',
      narration: 'Large per-request artifacts use compressed HTTP instead of crowding the control plane.',
      term: { word: 'Artifact plane', meaning: 'A second network path for bulk files, derived from the same controller host with a different port.' },
      points: [
        'Velo stays small: control messages and metric summaries.',
        'Bulk files use HTTP/1 + zstd so they do not clog the control plane.',
        'Synthetic / summary-only runs may never need this path.',
      ],
      caption: 'Small control on Velo. Big files on HTTP.',
    } as SlideDefinition,

    {
      eyebrow: 'Controller cost',
      title: 'Why spend a whole rank on a non-loading process?',
      lede: 'A dedicated controller rank keeps coordination, START sync, merging, and artifact handling away from measured load. That does not always mean a dedicated node — only a dedicated role.',
      narration: 'A dedicated controller rank keeps coordination responsive while cells generate maximum load.',
      term: { word: 'Dedicated rank', meaning: 'A process role, not necessarily a whole machine. Rank 0 can share a node with a cell if isolation is less important than allocation efficiency.' },
      points: [
        'Dedicated role: yes — keeps measurement clean and merge simple.',
        'Dedicated node: optional — the script defaults to one task per node.',
        'Co-locate controller + cell when machine count is scarce.',
      ],
      caption: 'Pay for coordination. Do not always pay for a whole idle node.',
    } as SlideDefinition,

    {
      eyebrow: 'Fan-out',
      title: 'Rank 0 fans distinct work slices out to the cells',
      lede: 'The controller begins with one global benchmark plan. As each cell registers over Velo, rank 0 replies with that cell\'s sliced envelope so the cells divide the work without overlap or omissions.',
      narration: 'The controller partitions one global plan into distinct slices and fans them out together.',
      term: { word: 'Fan-out', meaning: 'One source distributes different pieces of a larger job to several workers.' },
      points: [
        'The controller owns the one resolved Config v2 benchmark plan.',
        'Cell 0 gets slice 0, cell 1 gets slice 1, and so on.',
        'Together the slices tile the global request or conversation budget exactly.',
      ],
      caption: 'One benchmark plan fans out into disjoint cell-owned slices.',
    } as SlideDefinition,

    {
      eyebrow: 'Fan-in',
      title: 'Cells fan their finished results back into rank 0',
      lede: 'At the end, direction reverses. Every cell sends exactly one terminal partition or folded store over Velo. Rank 0 collects all expected children and merges them into the global result.',
      narration: 'Cells return their completed slices in parallel, and rank zero merges one final report.',
      term: { word: 'Fan-in', meaning: 'Several workers return their outputs to one collector, which combines them.' },
      points: [
        'Each cell ships one result partition or one folded metric store.',
        'Rank 0 waits for every expected cell before completing the merge.',
        'The merged output becomes one authoritative AIPerf report.',
      ],
      caption: 'Many cell results fan in to the original rank-0 controller.',
    } as SlideDefinition,

    {
      eyebrow: 'Try it',
      title: 'The two commands you actually type',
      lede: 'You rarely wire this up by hand. AIPerf generates a ready-to-submit SLURM script for you, and a single command runs the whole cellular benchmark inside the allocation.',
      narration: 'Generate the batch script, submit it, and AIPerf handles ranks, Velo, load, and results.',
      points: [
        'Generate a submission script from your benchmark config.',
        'Submit it with SLURM; every task launches the same run command.',
        'Rank assignment, Velo wiring, and merging all happen automatically.',
      ],
      caption: 'You describe the benchmark; AIPerf handles the cluster choreography.',
    } as SlideDefinition,
  ],
  scenesById: new Map(),
};

/**
 * Registry of all built-in explainer decks.
 * Add new decks here to make them available at runtime.
 */
export const BUILTIN_DECKS = [AIPERF_FLOW_SYSTEM_DECK, SLURM_VELO_DECK] as const;

/**
 * Load all built-in decks into the provided registry.
 * Each deck is pre-compiled with full metadata and narration.
 */
export function loadBuiltinDecks(registry: { register: (deck: ExplainerDefinition) => void }): void {
  for (const deck of BUILTIN_DECKS) {
    registry.register(deck);
  }
}
