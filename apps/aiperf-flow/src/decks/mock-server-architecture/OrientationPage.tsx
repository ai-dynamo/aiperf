/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Orientation chapter of the Mock Foundry deck — the process cutaway. Signature walkthrough is
//! the one-request-end-to-end journey through a single mock server process.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "orientation")!;

export function OrientationPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="aiperf-mock-server is launched independently and looks like an ordinary inference target to AIPerf. It depends on aiperf-runtime, but the product execution path never depends on the mock. One server process carries a request across parsing, token budgeting, latency, streaming, and accounting."
      signature={pageById("request-journey")}
      pages={pagesForChapter("orientation")}
    />
  );
}
