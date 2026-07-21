/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime-and-ingress chapter of the Mock Foundry deck — the listener + route manifold. Signature
//! walkthrough is the tuned Hyper TCP listener accepting a connection into the shared router.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "ingress")!;

export function IngressPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="Clap config definitions are authoritative but definition alone does not prove wiring. TCP, HTTP/1.1, h2c, UDS, and TLS listeners all converge on one Axum route surface where every HTTP dialect shares state while keeping distinct wire shapes."
      signature={pageById("tcp-listener")}
      pages={pagesForChapter("ingress")}
    />
  );
}
