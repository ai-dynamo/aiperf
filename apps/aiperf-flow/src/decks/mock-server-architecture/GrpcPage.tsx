/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! gRPC-and-Riva chapter of the Mock Foundry deck — the protobuf switching yard + transducers.
//! Signature walkthrough is the KServe unary ModelInfer tensor round-trip.

import { ChapterPage } from "./components.js";
import { CHAPTERS, pageById, pagesForChapter } from "./catalog.js";

const CHAPTER = CHAPTERS.find((c) => c.id === "grpc")!;

export function GrpcPage(): React.JSX.Element {
  return (
    <ChapterPage
      chapter={CHAPTER}
      lead="KServe unary and streaming ModelInfer, readiness services, and behavior-gated embedding/ranking/image tensors all ride one protobuf RPC. Riva ASR, TTS, and NLP are exposed only through gRPC service methods — there is no HTTP route for them in the mock router."
      signature={pageById("grpc-unary")}
      pages={pagesForChapter("grpc")}
    />
  );
}
