// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import type { Page } from "@playwright/test";

export const RUNTIME_PERFORMANCE_ENTRY_NAMES = Object.freeze({
  evaluation: "aiperf-flow:evaluation",
  draw: "aiperf-flow:draw",
  total: "aiperf-flow:total",
});

export type RuntimePerformanceEntryNames = Readonly<{
  evaluation: string;
  draw: string;
  total: string;
}>;

export type RuntimeFrameSample = Readonly<{
  evaluationMs: number;
  drawMs: number;
  totalMs: number;
}>;

export type RuntimeMeasurementInput = Readonly<{
  samples: readonly RuntimeFrameSample[];
}>;

export type CollectRuntimeMetricsOptions = Readonly<{
  entryNames?: RuntimePerformanceEntryNames;
  sampleCount?: number;
  timeoutMs?: number;
}>;

type PhaseDurations = Readonly<{
  evaluation: readonly number[];
  draw: readonly number[];
  total: readonly number[];
}>;

function assertSampleCount(sampleCount: number | undefined): void {
  if (
    sampleCount !== undefined &&
    (!Number.isSafeInteger(sampleCount) || sampleCount <= 0)
  ) {
    throw new RangeError("sampleCount must be a positive safe integer.");
  }
}

function assertDurations(
  phase: keyof PhaseDurations,
  durations: readonly number[],
): void {
  for (const duration of durations) {
    if (!Number.isFinite(duration) || duration < 0) {
      throw new Error(
        `Runtime ${phase} performance entry has invalid duration ${duration}.`,
      );
    }
  }
}

async function readPhaseDurations(
  page: Page,
  entryNames: RuntimePerformanceEntryNames,
): Promise<PhaseDurations> {
  return page.evaluate((names) => {
    const durations = (name: string): number[] =>
      performance
        .getEntriesByName(name, "measure")
        .map((entry) => entry.duration);

    return {
      evaluation: durations(names.evaluation),
      draw: durations(names.draw),
      total: durations(names.total),
    };
  }, entryNames);
}

/** Clears only AIPerf Flow measures so a collection window starts empty. */
export async function clearRuntimeMetrics(
  page: Page,
  entryNames: RuntimePerformanceEntryNames = RUNTIME_PERFORMANCE_ENTRY_NAMES,
): Promise<void> {
  await page.evaluate((names) => {
    performance.clearMeasures(names.evaluation);
    performance.clearMeasures(names.draw);
    performance.clearMeasures(names.total);
  }, entryNames);
}

/**
 * Collects real evaluator, draw, and total measures emitted by the browser.
 *
 * Call `clearRuntimeMetrics` before exercising the runtime when the page may
 * already contain startup frames.
 */
export async function collectRuntimeMetrics(
  page: Page,
  options: CollectRuntimeMetricsOptions = {},
): Promise<readonly RuntimeFrameSample[]> {
  const entryNames = options.entryNames ?? RUNTIME_PERFORMANCE_ENTRY_NAMES;
  const sampleCount = options.sampleCount;
  assertSampleCount(sampleCount);

  if (sampleCount !== undefined) {
    await page.waitForFunction(
      ({ names, count }) =>
        performance.getEntriesByName(names.evaluation, "measure").length >= count &&
        performance.getEntriesByName(names.draw, "measure").length >= count &&
        performance.getEntriesByName(names.total, "measure").length >= count,
      { names: entryNames, count: sampleCount },
      { timeout: options.timeoutMs },
    );
  }

  const phases = await readPhaseDurations(page, entryNames);
  assertDurations("evaluation", phases.evaluation);
  assertDurations("draw", phases.draw);
  assertDurations("total", phases.total);

  const availableCounts = [
    phases.evaluation.length,
    phases.draw.length,
    phases.total.length,
  ];
  if (availableCounts.some((count) => count === 0)) {
    throw new Error(
      `Runtime performance entries are missing: evaluation=${availableCounts[0]}, draw=${availableCounts[1]}, total=${availableCounts[2]}.`,
    );
  }
  if (new Set(availableCounts).size !== 1) {
    throw new Error(
      `Runtime performance entry counts do not match: evaluation=${availableCounts[0]}, draw=${availableCounts[1]}, total=${availableCounts[2]}.`,
    );
  }

  const count = sampleCount ?? availableCounts[0]!;
  return Object.freeze(
    Array.from({ length: count }, (_, index) =>
      Object.freeze({
        evaluationMs: phases.evaluation[index]!,
        drawMs: phases.draw[index]!,
        totalMs: phases.total[index]!,
      }),
    ),
  );
}

/** Formats samples for direct consumption by `measure-runtime.mjs`. */
export function formatRuntimeMetricsJson(
  samples: readonly RuntimeFrameSample[],
): string {
  if (samples.length === 0) {
    throw new Error("Runtime metrics JSON requires at least one frame sample.");
  }
  for (const sample of samples) {
    assertDurations("evaluation", [sample.evaluationMs]);
    assertDurations("draw", [sample.drawMs]);
    assertDurations("total", [sample.totalMs]);
  }
  const input: RuntimeMeasurementInput = { samples };
  return `${JSON.stringify(input, null, 2)}\n`;
}

/** Writes deterministic measurement input JSON for `measure-runtime.mjs`. */
export async function writeRuntimeMetricsJson(
  path: string,
  samples: readonly RuntimeFrameSample[],
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, formatRuntimeMetricsJson(samples), "utf8");
}
