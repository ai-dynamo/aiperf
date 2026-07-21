/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLM-protocols chapter of the Mock Foundry deck — the transparent protocol pipes. Signature
//! walkthrough is SSE stream assembly: generated token events precede terminal usage and [DONE].

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "llm")!;

export function LlmProtocolsPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="OpenAI chat and completions, Anthropic Messages, the OpenAI Responses API, reasoning content, and token-native vLLM generate all share one generation seam but keep provider-specific wire shapes. Usage is terminal accounting, never a generated-token timing sample."
      signature={pageById("sse")}
      pages={pagesForChapter("llm")}
    />
  );
}
