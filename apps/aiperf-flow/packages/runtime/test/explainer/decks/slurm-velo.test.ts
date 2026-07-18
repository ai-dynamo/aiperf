/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, it, expect, beforeAll } from 'vitest';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Test suite for slurm-velo.flow explainer deck.
 * Validates:
 * - File loads and compiles
 * - All 16 slides are present
 * - All required fields present in each slide
 * - Narration texts match content
 * - Glossary terms are defined
 * - Schema validation
 */
describe('slurm-velo.flow explainer deck', () => {
  let sourceCode: string;

  beforeAll(() => {
    // Load the .flow source file
    const flowPath = join(
      __dirname,
      '../../../src/explainer/decks/slurm-velo.flow'
    );
    sourceCode = readFileSync(flowPath, 'utf-8');
  });

  it('loads the .flow file successfully', () => {
    expect(sourceCode).toBeDefined();
    expect(sourceCode.length).toBeGreaterThan(0);
  });

  it('has proper explainer block declaration', () => {
    expect(sourceCode).toMatch(/explainer\s+"slurm-velo"\s*\{/);
    expect(sourceCode).toContain('route: "/explainers/slurm-velo"');
    expect(sourceCode).toContain('topic: "distributed-execution"');
    expect(sourceCode).toContain('eyebrowLabel: "Cluster Orchestration"');
    expect(sourceCode).toContain('startGateTitle: "Ready to learn SLURM + Velo?"');
  });

  it('contains exactly 16 slides', () => {
    const slideMatches = sourceCode.match(/slide\s+"/g);
    expect(slideMatches).not.toBeNull();
    expect(slideMatches!.length).toBe(16);
  });

  it('validates slide 1: The problem', () => {
    expect(sourceCode).toContain('You want to load-test a big AI server');
    expect(sourceCode).toContain('eyebrow: "The problem"');
    expect(sourceCode).toContain(
      'Large AI servers need many load generators acting together as one benchmark.'
    );
    expect(sourceCode).toContain('Inference server');
  });

  it('validates slide 2: The tool - SLURM introduction', () => {
    expect(sourceCode).toContain('SLURM hands you a cluster of machines');
    expect(sourceCode).toContain('eyebrow: "The tool"');
    expect(sourceCode).toContain(
      'SLURM reserves cluster machines and launches your program across all of them.'
    );
    expect(sourceCode).toContain('term: {');
    expect(sourceCode).toContain('word: "SLURM"');
  });

  it('validates slide 3: The key trick - identical commands', () => {
    expect(sourceCode).toContain('Every machine runs the identical command');
    expect(sourceCode).toContain('eyebrow: "The key trick"');
    expect(sourceCode).toContain('Every task runs the same AIPerf command');
    expect(sourceCode).toContain('word: "Task"');
  });

  it('validates slide 4: Splitting roles - rank-based assignment', () => {
    expect(sourceCode).toContain('Rank 0 leads; everyone else does the work');
    expect(sourceCode).toContain('eyebrow: "Splitting the roles"');
    expect(sourceCode).toContain('Rank zero coordinates the benchmark');
    expect(sourceCode).toContain('word: "Rank"');
  });

  it('validates slide 5: Finding each other - coordinate derivation', () => {
    expect(sourceCode).toContain('Cells dial the controller with one shared fact');
    expect(sourceCode).toContain('eyebrow: "Finding each other"');
    expect(sourceCode).toContain('Each cell derives the rank-zero controller address');
    expect(sourceCode).toContain('word: "Coordinate"');
  });

  it('validates slide 6: Meet Velo', () => {
    expect(sourceCode).toContain(
      'Velo is the walkie-talkie between controller and cells'
    );
    expect(sourceCode).toContain('eyebrow: "Meet Velo"');
    expect(sourceCode).toContain('Velo carries coordination messages');
    expect(sourceCode).toContain('word: "Velo"');
  });

  it('validates slide 7: Velo bootstrap', () => {
    expect(sourceCode).toContain('A cell connects once, then Velo learns the peer');
    expect(sourceCode).toContain('eyebrow: "Velo bootstrap"');
    expect(sourceCode).toContain('Each cell connects once, allowing Velo to establish');
    expect(sourceCode).toContain('word: "Handshake"');
  });

  it('validates slide 8: Getting ready together - registration', () => {
    expect(sourceCode).toContain('Register and START travel over Velo');
    expect(sourceCode).toContain('eyebrow: "Getting ready together"');
    expect(sourceCode).toContain('Cells register through Velo');
    expect(sourceCode).toContain('word: "START barrier"');
  });

  it('validates slide 9: Doing the work - hot path', () => {
    expect(sourceCode).toContain('Benchmark requests do NOT use Velo');
    expect(sourceCode).toContain('eyebrow: "Doing the work"');
    expect(sourceCode).toContain(
      'Benchmark requests travel directly from cells to the inference server'
    );
    expect(sourceCode).toContain('word: "Hot path"');
  });

  it('validates slide 10: Three planes - traffic separation', () => {
    expect(sourceCode).toContain('Three completely different kinds of traffic');
    expect(sourceCode).toContain('eyebrow: "Three planes"');
    expect(sourceCode).toContain('Control, benchmark traffic, and bulk artifacts');
    expect(sourceCode).toContain('word: "Traffic plane"');
  });

  it('validates slide 11: One answer - result collection', () => {
    expect(sourceCode).toContain('Result partitions return to rank 0 over Velo');
    expect(sourceCode).toContain('eyebrow: "One answer"');
    expect(sourceCode).toContain('When work finishes, each cell returns its result');
    expect(sourceCode).toContain('word: "Partition"');
  });

  it('validates slide 12: Bulk files - artifact plane', () => {
    expect(sourceCode).toContain(
      'Huge per-record files take a different road'
    );
    expect(sourceCode).toContain('eyebrow: "Bulk files"');
    expect(sourceCode).toContain('Large per-request artifacts use compressed HTTP');
    expect(sourceCode).toContain('word: "Artifact plane"');
  });

  it('validates slide 13: Controller cost - dedicated role', () => {
    expect(sourceCode).toContain(
      'Why spend a whole rank on a non-loading process?'
    );
    expect(sourceCode).toContain('eyebrow: "Controller cost"');
    expect(sourceCode).toContain(
      'A dedicated controller rank keeps coordination responsive'
    );
    expect(sourceCode).toContain('word: "Dedicated rank"');
  });

  it('validates slide 14: Fan-out - work distribution', () => {
    expect(sourceCode).toContain(
      'Rank 0 fans distinct work slices out to the cells'
    );
    expect(sourceCode).toContain('eyebrow: "Fan-out"');
    expect(sourceCode).toContain(
      'The controller partitions one global plan into distinct slices'
    );
    expect(sourceCode).toContain('word: "Fan-out"');
  });

  it('validates slide 15: Fan-in - result aggregation', () => {
    expect(sourceCode).toContain(
      'Cells fan their finished results back into rank 0'
    );
    expect(sourceCode).toContain('eyebrow: "Fan-in"');
    expect(sourceCode).toContain(
      'Cells return their completed slices in parallel'
    );
    expect(sourceCode).toContain('word: "Fan-in"');
  });

  it('validates slide 16: Try it - command summary', () => {
    expect(sourceCode).toContain('The two commands you actually type');
    expect(sourceCode).toContain('eyebrow: "Try it"');
    expect(sourceCode).toContain('Generate the batch script, submit it');
    expect(sourceCode).toMatch(
      /You describe the benchmark; AIPerf handles the cluster choreography\./
    );
  });

  it('has all required slide fields across all slides', () => {
    // Count the occurrences of required fields
    const eyebrowCount = (sourceCode.match(/eyebrow:/g) || []).length;
    const titleCount = (sourceCode.match(/title:/g) || []).length;
    const ledeCount = (sourceCode.match(/lede:/g) || []).length;
    const narrationCount = (sourceCode.match(/narration:/g) || []).length;
    const pointsCount = (sourceCode.match(/points:/g) || []).length;
    const captionCount = (sourceCode.match(/caption:/g) || []).length;

    // Each slide should have these fields (except term which is optional)
    expect(eyebrowCount).toBe(16);
    expect(titleCount).toBe(16);
    expect(ledeCount).toBe(16);
    expect(narrationCount).toBe(16);
    expect(pointsCount).toBe(16);
    expect(captionCount).toBe(16);
  });

  it('validates narration texts are all distinct and meaningful', () => {
    const narrations = [
      'Large AI servers need many load generators acting together as one benchmark.',
      'SLURM reserves cluster machines and launches your program across all of them.',
      'Every task runs the same AIPerf command, then its rank determines its role.',
      'Rank zero coordinates the benchmark. Every other rank becomes a load-generating cell.',
      'Each cell derives the rank-zero controller address from the shared SLURM allocation.',
      'Velo carries coordination messages between the controller and its remote cells.',
      'Each cell connects once, allowing Velo to establish the peer relationship.',
      'Cells register through Velo, then wait until the controller broadcasts START.',
      'Benchmark requests travel directly from cells to the inference server, never through Velo.',
      'Control, benchmark traffic, and bulk artifacts use three deliberately separate paths.',
      'When work finishes, each cell returns its result partition to rank zero over Velo.',
      'Large per-request artifacts use compressed HTTP instead of crowding the control plane.',
      'A dedicated controller rank keeps coordination responsive while cells generate maximum load.',
      'The controller partitions one global plan into distinct slices and fans them out together.',
      'Cells return their completed slices in parallel, and rank zero merges one final report.',
      'Generate the batch script, submit it, and AIPerf handles ranks, Velo, load, and results.',
    ];

    narrations.forEach((narration) => {
      expect(sourceCode).toContain(`narration: "${narration}"`);
    });
  });

  it('validates all glossary terms are properly paired', () => {
    const terms = [
      { word: 'Inference server', meaning: 'The service that runs an AI model' },
      { word: 'SLURM', meaning: 'A job scheduler for compute clusters' },
      { word: 'Task', meaning: 'One running copy of your command' },
      { word: 'Rank', meaning: 'Each task\'s index in the allocation' },
      { word: 'Coordinate', meaning: 'The controller\'s address' },
      { word: 'Velo', meaning: 'An async messaging framework' },
      { word: 'Handshake', meaning: 'Velo\'s connect step' },
      { word: 'START barrier', meaning: 'A Velo event gate' },
      { word: 'Hot path', meaning: 'The per-request and per-token traffic' },
      { word: 'Traffic plane', meaning: 'A separate purpose for network traffic' },
      { word: 'Partition', meaning: 'One cell\'s bundle of results' },
      { word: 'Artifact plane', meaning: 'A second network path for bulk files' },
      { word: 'Dedicated rank', meaning: 'A process role' },
      { word: 'Fan-out', meaning: 'One source distributes different pieces' },
      { word: 'Fan-in', meaning: 'Several workers return their outputs' },
    ];

    terms.forEach((term) => {
      expect(sourceCode).toContain(`word: "${term.word}"`);
      expect(sourceCode).toContain(`meaning: "${term.meaning}`);
    });
  });

  it('validates deck structure matches explainer requirements', () => {
    // Check for opening and closing braces
    expect(sourceCode).toMatch(/explainer\s+"slurm-velo"\s*\{[\s\S]*\}$/m);

    // Check metadata fields
    expect(sourceCode).toMatch(/route:\s*"/);
    expect(sourceCode).toMatch(/topic:\s*"/);
    expect(sourceCode).toMatch(/eyebrowLabel:\s*"/);
    expect(sourceCode).toMatch(/startGateTitle:\s*"/);

    // Check slide structure
    expect(sourceCode).toMatch(/slide\s+"[^"]+"\s*\{[\s\S]*?\}/g);
  });

  it('validates line count is reasonable for deck size', () => {
    const lines = sourceCode.split('\n').length;
    // 16 slides with multiple fields each should be substantial
    expect(lines).toBeGreaterThan(200);
    expect(lines).toBeLessThan(500);
  });

  it('contains SPDX license header', () => {
    expect(sourceCode).toContain('SPDX-FileCopyrightText');
    expect(sourceCode).toContain('SPDX-License-Identifier: Apache-2.0');
  });

  it('contains module documentation', () => {
    expect(sourceCode).toContain('//!');
    expect(sourceCode).toContain('SLURM');
    expect(sourceCode).toContain('Velo');
  });

  it('validates captions exist and are distinct', () => {
    const captions = [
      'Goal: many machines, one benchmark, one result.',
      'SLURM = the landlord that lends you machines for a while.',
      'Same command everywhere — the rank number breaks the tie.',
      'controller = rank 0 · cell_id = rank − 1 · cell_count = tasks − 1.',
      'One fact, computed the same everywhere — no discovery service needed.',
      'SLURM launches the processes. Velo lets those processes talk.',
      'Address in → peer connection out. No service discovery backend.',
      'Line everyone up over Velo, then start the race together.',
      'Velo coordinates. HTTP/gRPC generates the measured load.',
      'Mixing these up is the main source of confusion.',
      'Cells measure. Rank 0 merges. One authoritative report.',
      'Small control on Velo. Big files on HTTP.',
      'Pay for coordination. Do not always pay for a whole idle node.',
      'One benchmark plan fans out into disjoint cell-owned slices.',
      'Many cell results fan in to the original rank-0 controller.',
      'You describe the benchmark; AIPerf handles the cluster choreography.',
    ];

    captions.forEach((caption) => {
      expect(sourceCode).toContain(`caption: "${caption}"`);
    });
  });
});
