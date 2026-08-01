/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { defineConfig, devices } from "@playwright/test";

const PORT = 4319;

/**
 * Browser checks run against a production build served by `vite preview`, never the dev server:
 * HMR remounts the deck shell mid-run, which restarts the reveal cascade and makes any
 * settle-then-measure assertion flaky.
 */
export default defineConfig({
  testDir: "./e2e",
  // A deck test walks every slide and waits for that slide's reveal cascade to stop moving. The
  // cascade itself is real animation time the test cannot skip: `useReveal` steps 220ms per node,
  // so the 27-slide deck (240 nodes) spends (240 - 27) * 220ms ~= 47s just revealing, plus the
  // per-slide fitView animation and settle window. 120s leaves room over that ~70s worst case.
  timeout: 120_000,
  // These assert layout geometry, so a second worker resizing nothing is still fine — but a
  // single browser keeps the screenshots deterministic and the run short.
  workers: 1,
  fullyParallel: false,
  reporter: process.env.CI === undefined ? [["list"]] : [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    // 1600x900 is the framing these decks are authored against; the widest node is ~1126 CSS px.
    viewport: { width: 1600, height: 900 },
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run build && npm run preview -- --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}`,
    // Never reuse a server that is already up, even locally. These checks compare a node's
    // declared box against what it actually renders, so serving a stale `dist/` would report a
    // fix as landed when it had not been rebuilt — a failure mode this suite exists to prevent.
    // With `strictPort`, a leftover preview now fails loudly instead of being silently trusted.
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
