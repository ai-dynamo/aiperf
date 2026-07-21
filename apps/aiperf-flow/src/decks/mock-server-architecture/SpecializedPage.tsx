/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Specialized-endpoints chapter of the Mock Foundry deck — the looms, sorters, and chambers.
//! Signature walkthrough is RAG/KServe HTTP aliases resolving to shared handlers.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "specialized")!;

export function SpecializedPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="Embeddings, ranking dialects, TGI generate, image generation/edit/retrieval, multimodal payloads, and RAG/KServe HTTP aliases are all deterministic for the same input and reuse the shared response machinery rather than forking it."
      signature={pageById("rag-kserve-http")}
      pages={pagesForChapter("specialized")}
    />
  );
}
