/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Proof-and-boundaries chapter of the Mock Foundry deck — the exploded evidence machine.
//! Signature walkthrough is the implementation-to-proof evidence graph.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "proof")!;

export function ProofPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="Raw-record e2e is the strongest evidence, then integration, unit, and implementation-only. Boundaries are explicit: multi-process is TCP/HTTP-only and Riva is gRPC-only. Every atlas feature links to its implementation and its strongest available proof."
      signature={pageById("proof-graph")}
      pages={pagesForChapter("proof")}
    />
  );
}
