/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scheduler-and-cache chapter of the Mock Foundry deck — the conveyor + cache library. Signature
//! walkthrough is prefill/decode stepping under configured capacities.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "scheduler")!;

export function SchedulerPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="Scheduler ticks admit prefill work and emit decode tokens under configured capacities; the maximum batch size creates a visible admission and throughput knee, and configured collapse reduces service rate after saturation. Prompt token blocks hash into a bounded prefix cache whose latency effect is opt-in."
      signature={pageById("prefill-decode")}
      pages={pagesForChapter("scheduler")}
    />
  );
}
