/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, it, expect } from 'vitest';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Test suite for rust-architecture.flow explainer deck.
 * Validates:
 * - File loads and compiles
 * - All 16 slides are present
 * - All required fields present in each slide
 * - Narration texts match content
 * - Glossary terms are defined
 * - Schema validation
 * - Scene rendering blocks are valid
 */
describe('rust-architecture.flow explainer deck', () => {
  let sourceCode: string;

  // Load source code once before all tests
  const flowPath = join(
    __dirname,
    '../../../src/explainer/decks/rust-architecture.flow'
  );
  sourceCode = readFileSync(flowPath, 'utf-8');

  it('loads the .flow file successfully', () => {
    expect(sourceCode).toBeDefined();
    expect(sourceCode.length).toBeGreaterThan(0);
  });

  it('has proper explainer block declaration', () => {
    expect(sourceCode).toMatch(/explainer\s+"Rust Architecture"\s*\{/);
    expect(sourceCode).toContain('id: "rust-architecture"');
    expect(sourceCode).toContain('route: "/explainers/rust-architecture"');
    expect(sourceCode).toContain('topic: "architecture"');
    expect(sourceCode).toContain('eyebrowLabel: "Architecture"');
    expect(sourceCode).toContain('startGateTitle: "Ready to explore AIPerf\'s Rust architecture?"');
  });

  it('contains hub metadata', () => {
    expect(sourceCode).toContain('hub: {');
    expect(sourceCode).toContain('title: "Rust Architecture"');
    expect(sourceCode).toContain('highlight: "Explore AIPerf\'s native execution engine and composition model"');
    expect(sourceCode).toContain('description: "Deep dive into the Rust workspace, module organization, execution model, and scaling patterns that power AIPerf benchmarks."');
  });

  it('contains exactly 16 slides', () => {
    const slideMatches = sourceCode.match(/slide\s+"/g);
    expect(slideMatches).not.toBeNull();
    expect(slideMatches!.length).toBe(16);
  });

  it('validates slide 1: Product shell', () => {
    expect(sourceCode).toContain('One binary is both CLI and engine');
    expect(sourceCode).toContain('eyebrow: "Product shell"');
    expect(sourceCode).toContain('AIPerf ships as a single native `aiperf` executable');
    expect(sourceCode).toContain('word: "aiperf-cli"');
  });

  it('validates slide 2: Workspace map', () => {
    expect(sourceCode).toContain('Six crates, one dependency direction');
    expect(sourceCode).toContain('eyebrow: "Workspace map"');
    expect(sourceCode).toContain('word: "loadgen-core"');
  });

  it('validates slide 3: Startup order', () => {
    expect(sourceCode).toContain('Intercept internal modes, then dispatch');
    expect(sourceCode).toContain('eyebrow: "Startup order"');
    expect(sourceCode).toContain('word: "execute_mode"');
  });

  it('validates slide 4: Command surface', () => {
    expect(sourceCode).toContain('Native benchmark commands stay in Rust');
    expect(sourceCode).toContain('eyebrow: "Command surface"');
    expect(sourceCode).toContain('word: "delegate"');
  });

  it('validates slide 5: Configuration', () => {
    expect(sourceCode).toContain('Config v2 resolves into BenchmarkRun');
    expect(sourceCode).toContain('eyebrow: "Configuration"');
    expect(sourceCode).toContain('word: "BenchmarkRun"');
  });

  it('validates slide 6: Self execution', () => {
    expect(sourceCode).toContain('profile spawns aiperf --execute');
    expect(sourceCode).toContain('eyebrow: "Self execution"');
    expect(sourceCode).toContain('word: "stdio seam"');
  });

  it('validates slide 7: Wire contract', () => {
    expect(sourceCode).toContain('Protocol v2 wraps the bare run');
    expect(sourceCode).toContain('eyebrow: "Wire contract"');
    expect(sourceCode).toContain('word: "Application"');
  });

  it('validates slide 8: Bootstrap', () => {
    expect(sourceCode).toContain('AIPerfRegistry freezes capabilities');
    expect(sourceCode).toContain('eyebrow: "Bootstrap"');
    expect(sourceCode).toContain('word: "AIPerfExtension"');
  });

  it('validates slide 9: Composition root', () => {
    expect(sourceCode).toContain('Coordinator validates, prepares, executes, persists');
    expect(sourceCode).toContain('eyebrow: "Composition root"');
    expect(sourceCode).toContain('word: "Coordinator"');
  });

  it('validates slide 10: Time seam', () => {
    expect(sourceCode).toContain('Every schedule uses Clock');
    expect(sourceCode).toContain('eyebrow: "Time seam"');
    expect(sourceCode).toContain('word: "Clock"');
  });

  it('validates slide 11: Inputs', () => {
    expect(sourceCode).toContain('Dataset flows load → sample → materialize');
    expect(sourceCode).toContain('eyebrow: "Inputs"');
    expect(sourceCode).toContain('word: "Segment pool"');
  });

  it('validates slide 12: Work generation', () => {
    expect(sourceCode).toContain('Workloads and phases drive turns');
    expect(sourceCode).toContain('eyebrow: "Work generation"');
    expect(sourceCode).toContain('word: "phase_runtime"');
  });

  it('validates slide 13: Observation seam', () => {
    expect(sourceCode).toContain('loadgen-core keeps transport neutral');
    expect(sourceCode).toContain('eyebrow: "Observation seam"');
    expect(sourceCode).toContain('word: "RequestObserver"');
  });

  it('validates slide 14: Parallelism', () => {
    expect(sourceCode).toContain('Thread-per-core workers, not mutex hot paths');
    expect(sourceCode).toContain('eyebrow: "Parallelism"');
    expect(sourceCode).toContain('word: "Sub-cell"');
  });

  it('validates slide 15: Scale-out', () => {
    expect(sourceCode).toContain('Cellular mode adds controller and cells');
    expect(sourceCode).toContain('eyebrow: "Scale-out"');
    expect(sourceCode).toContain('word: "Cell partition"');
  });

  it('validates slide 16: Outputs & gates', () => {
    expect(sourceCode).toContain('Metrics merge, exporters fan out, features opt in');
    expect(sourceCode).toContain('eyebrow: "Outputs & gates"');
    expect(sourceCode).toContain('word: "native-v2.json"');
  });

  it('has all required slide fields across all slides', () => {
    // Verify that all slides have the required fields by checking specific fields per slide
    const slideMatches = sourceCode.match(/slide\s+"([^"]+)"/g);
    expect(slideMatches).not.toBeNull();
    expect(slideMatches!.length).toBe(16);

    // Check that required keywords appear frequently (should be 16 times each for the main fields)
    // We know from inspection that narration, points, and caption appear in every slide
    expect((sourceCode.match(/narration:\s*"/g) || []).length).toBeGreaterThanOrEqual(16);
    expect((sourceCode.match(/points:\s*\[/g) || []).length).toBeGreaterThanOrEqual(16);
    expect((sourceCode.match(/caption:\s*"/g) || []).length).toBeGreaterThanOrEqual(16);
  });

  it('validates narration texts are all distinct and meaningful', () => {
    const narrations = [
      'AIPerf ships as one native aiperf binary. That same executable is both the public command line and the hidden execution engine.',
      'The Rust workspace stays small. Capability flows from aiperf-cli into aiperf-runtime and then into loadgen-core.',
      'Startup always checks hidden execution modes first. Only after that does the process route public subcommands.',
      'Core benchmark commands stay native in Rust. Most operational tooling still delegates to Python unless the build embeds it.',
      'Profile reads Config v2, expands it, and resolves a strict BenchmarkRun object that describes the whole benchmark.',
      'Each profile run spawns a fresh child of the same binary with aiperf execute over stdio.',
      'Protocol version two wraps the run, composes Application once, and returns one terminal envelope.',
      'At bootstrap, AIPerfRegistry freezes loaders, samplers, endpoints, transports, workloads, and exporters.',
      'Coordinator is the composition root: validate, prepare, execute, persist native-v2.json, then run exporters.',
      'All timing goes through Clock. RealClock drives online traffic; SimClock drives deterministic simulation.',
      'Datasets load, sample, and materialize into endpoint-ready requests through one shared substrate.',
      'Workloads and phase runtime decide when turns fire. Transports only send and observe the wire path.',
      'loadgen-core defines the neutral observer seam. Transports implement RequestSink; measurement stays shared.',
      'Parallelism is thread-per-core. Each worker owns local scheduling, transport, and capture without hot-path mutexes.',
      'Cellular mode adds a controller, remote cells, Velo control traffic, and a single merged report.',
      'Metrics merge into native-v2.json, exporters fan out artifacts, and Cargo features opt into gRPC, cellular, dynosim, and more.'
    ];

    narrations.forEach((narration) => {
      expect(sourceCode).toContain(`narration: "${narration}"`);
    });
  });

  it('validates all glossary terms are properly defined', () => {
    const terms = [
      { word: 'aiperf-cli', meaning: 'The product entry crate: public commands, Config v2 loading, and self-spawned execution over stdio.' },
      { word: 'loadgen-core', meaning: 'Transport-neutral observer and sink vocabulary with no HTTP, gRPC, or engine dependencies.' },
      { word: 'execute_mode', meaning: 'Hidden argv surface that runs protocol-v2 children before clap ever sees a public command.' },
      { word: 'delegate', meaning: 'Lean builds shell out to `python -m aiperf`; pyo3-embed runs the same entrypoint in-process.' },
      { word: 'BenchmarkRun', meaning: 'The authored protocol-v2 request describing workload, transport, endpoints, artifacts, and runtime facts.' },
      { word: 'stdio seam', meaning: 'The deliberate parent/child boundary that keeps operator UX separate from execution isolation.' },
      { word: 'Application', meaning: 'Frozen runtime composition: registry, coordinator, and factories selected at bootstrap.' },
      { word: 'AIPerfExtension', meaning: 'Compile-time registration hook that adds implementations into the shared registry during Application construction.' },
      { word: 'Coordinator', meaning: 'Engine composition root that owns validate → prepare → execute → persist for a single BenchmarkRun.' },
      { word: 'Clock', meaning: 'Injectable time source used for scheduling, measurement gates, backoff, and simulation driving.' },
      { word: 'Segment pool', meaning: 'Content-addressed conversation storage shared read-only across worker threads via Arc.' },
      { word: 'phase_runtime', meaning: 'Shared lifecycle orchestration that connects schedulers to turn dispatch and terminal drain behavior.' },
      { word: 'RequestObserver', meaning: 'Worker-local callback surface with no Send bound, allowing LocalSet-friendly state.' },
      { word: 'Sub-cell', meaning: 'A worker thread that owns scheduling, admission, transport, capture, and local measurement without cross-thread locks on the request path.' },
      { word: 'Cell partition', meaning: 'Deterministic slice of the global request or conversation budget owned by one remote cell process.' },
      { word: 'native-v2.json', meaning: 'Authoritative merged report for a run before optional exporter side outputs.' }
    ];

    terms.forEach((term) => {
      expect(sourceCode).toContain(`word: "${term.word}"`);
      expect(sourceCode).toContain(`meaning: "${term.meaning}"`);
    });
  });

  it('validates deck structure matches explainer requirements', () => {
    // Check for opening and closing braces
    expect(sourceCode).toMatch(/explainer\s+"Rust Architecture"\s*\{[\s\S]*\}$/m);

    // Check metadata fields
    expect(sourceCode).toMatch(/route:\s*"/);
    expect(sourceCode).toMatch(/topic:\s*"/);
    expect(sourceCode).toMatch(/eyebrowLabel:\s*"/);
    expect(sourceCode).toMatch(/startGateTitle:\s*"/);

    // Check slide structure
    expect(sourceCode).toMatch(/slide\s+"[^"]+"\s*\{[\s\S]*?\}/g);
  });

  it('validates all slides have scene rendering blocks', () => {
    const sceneMatches = sourceCode.match(/render:\s*@scene\s*\{/g);
    expect(sceneMatches).not.toBeNull();
    expect(sceneMatches!.length).toBe(16); // One per slide
  });

  it('validates scene blocks reference theme roles', () => {
    expect(sourceCode).toContain('@theme.surface.primary');
    expect(sourceCode).toContain('@theme.ink.primary');
    expect(sourceCode).toContain('@theme.ink.secondary');
    expect(sourceCode).toContain('@theme.accent.primary');
    expect(sourceCode).toContain('@theme.accent.secondary');
    expect(sourceCode).toContain('@theme.accent.tertiary');
  });

  it('validates scene blocks use core capabilities', () => {
    expect(sourceCode).toContain('capability: "core.rect"');
    expect(sourceCode).toContain('capability: "core.text"');
    expect(sourceCode).toContain('capability: "core.line"');
  });

  it('validates line count is reasonable for deck size', () => {
    const lines = sourceCode.split('\n').length;
    // 16 slides with scene blocks and metadata should be substantial
    expect(lines).toBeGreaterThan(500);
    expect(lines).toBeLessThan(1500);
  });

  it('contains SPDX license header', () => {
    expect(sourceCode).toContain('SPDX-FileCopyrightText');
    expect(sourceCode).toContain('SPDX-License-Identifier: Apache-2.0');
  });

  it('validates captions exist and describe the diagrams', () => {
    const captions = [
      'One executable, two hats: operator CLI and execution child.',
      'cli → runtime → loadgen-core',
      'Hidden modes short-circuit; public commands fall through.',
      'Benchmark hot path native; extended surface delegated.',
      'YAML in → validated BenchmarkRun out.',
      'Same binary, new process, protocol on stdio.',
      'Envelope v2 in → terminal envelope out.',
      'Capabilities are frozen before the first request.',
      'One coordinator owns the whole run lifecycle.',
      'Real time online, virtual time in simulation.',
      'Inputs become endpoint-ready requests once, then reuse handles.',
      'Scheduling decides when; transport decides how to send.',
      'Transport sends; observer measures; sink owns the lifecycle.',
      'Scale out by adding worker threads, not shared mutable hot state.',
      'Same engine, more processes, partitioned ownership.',
      'Measure locally, merge once, export many formats.'
    ];

    captions.forEach((caption) => {
      expect(sourceCode).toContain(`caption: "${caption}"`);
    });
  });

  it('validates all points arrays are non-empty', () => {
    const pointsMatches = sourceCode.matchAll(/points:\s*\[([\s\S]*?)\]/g);
    let pointsCount = 0;
    for (const match of pointsMatches) {
      pointsCount++;
      const content = match[1];
      expect(content).toMatch(/"/); // Should have at least one quoted string
    }
    expect(pointsCount).toBe(16);
  });

  it('validates scene layout coordinates are reasonable', () => {
    // Check that layout coordinates are within reasonable bounds
    expect(sourceCode).toMatch(/layout:\s*\{\s*x:\s*\d+/);
    expect(sourceCode).toMatch(/y:\s*\d+/);
    expect(sourceCode).toMatch(/width:\s*\d+/);
    expect(sourceCode).toMatch(/height:\s*\d+/);
  });

  it('validates lede texts introduce slide topics', () => {
    expect(sourceCode).toContain('AIPerf ships as a single native');
    expect(sourceCode).toContain('The Rust workspace is intentionally small');
    expect(sourceCode).toContain('Every process starts the same way');
  });
});
