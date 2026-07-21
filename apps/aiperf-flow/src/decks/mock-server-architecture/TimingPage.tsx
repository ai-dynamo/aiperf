/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Timing-and-generation chapter of the Mock Foundry deck — the TTFT/ITL escapement. Signature
//! walkthrough is the independently paced first-token delay and generated-token gaps.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "timing")!;

export function TimingPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="Rust token generation is character/corpus based — it does not load a Hugging Face tokenizer, and the HF identity flags are unwired. Seeded inputs yield repeatable budgets; TTFT and ITL are independently paced and seeded jitter is reproducible through RealClock timerfd precision."
      signature={pageById("ttft-itl")}
      pages={pagesForChapter("timing")}
    />
  );
}
