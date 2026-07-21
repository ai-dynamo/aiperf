/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Faults-and-semantics chapter of the Mock Foundry deck — the injectors + verdict gates.
//! Signature walkthrough is a mid-stream SSE failure preserving partial-response evidence.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "semantics")!;

export function SemanticsPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="Seeded status injection chooses only from the configured status menu, Retry-After accompanies retryable responses, and a stream can fail after generated output while preserving partial-response evidence. Extended usage fields keep provider-specific names, and seeded accuracy verdicts feed a live oracle."
      signature={pageById("midstream")}
      pages={pagesForChapter("semantics")}
    />
  );
}
