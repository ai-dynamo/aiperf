/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Observability-and-deployment chapter of the Mock Foundry deck — the telemetry deck + replicated
//! foundries. Signature walkthrough is throughput-linked synthetic GPU load.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "observability")!;

export function ObservabilityPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="One Prometheus endpoint can emit vLLM, SGLang, TRT-LLM, or Dynamo dialects; synthetic DCGM telemetry is deterministic under a seed and follows observed request throughput. The L4 balancer round-robins TCP across isolated child servers — but multi-process skips gRPC and UDS, and --access-logs is defined yet unwired."
      signature={pageById("throughput-load")}
      pages={pagesForChapter("observability")}
    />
  );
}
