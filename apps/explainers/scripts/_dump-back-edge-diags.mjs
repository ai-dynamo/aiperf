/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { resolve } from "node:path";

const { stdout } = await promisify(execFile)(
  "npx",
  ["vite-node", resolve("scripts/compile-decks.ts")],
  { cwd: process.cwd(), maxBuffer: 64 * 1024 * 1024 },
);
const bundle = JSON.parse(stdout);
const slides = bundle.resolvedScenes.filter(
  (s) =>
    s.deckId === "sdk-diagram-catalog" &&
    (s.slideId === "sdk.retry" || s.slideId === "sdk.loop"),
);
for (const s of slides) {
  const all = s.snapshot.diagnostics ?? [];
  const diags = all.filter(
    (d) =>
      (d.nodeIds ?? []).includes("hero__back-edge") ||
      String(d.message ?? "").includes("hero__back-edge") ||
      String(d.code ?? "").includes("CONNECTOR"),
  );
  console.log(
    s.slideId,
    "total",
    all.length,
    "filtered",
    diags.length,
    "connector?",
    (s.snapshot.connectors ?? []).some((c) => c.id === "hero__back-edge"),
  );
  for (const d of diags) {
    console.log(" ", d.severity, d.code, d.message, d.nodeIds);
  }
}
