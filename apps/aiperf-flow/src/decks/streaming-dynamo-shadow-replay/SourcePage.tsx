/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, DbNode, MiniArrow } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

//! Source page: S3 and local source factories, Dynamo format decoder, checkpoint backend.

/** Source layer: acquire and decode trace files. */
export function SourcePage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Acquire and decode trace files">
        Every trace file is one immutable partition. The source factory discovers partitions — by paginating an S3
        bucket or scanning a local path — assigns each a dense position, and hands it to the format decoder. The
        decoder decompresses gzip and parses JSONL into typed request units. A checkpoint backend persists the per-partition
        read cursor so the source can resume from exactly where it left off after a restart.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "STREAMING · SOURCE LAYER",
          title: "How do trace files become request units?",
          body: "Discover → acquire → decode → checkpoint. Each file is one immutable partition with a stable content digest.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "S3 source",
            diagram: (
              <Diagram>
                <DbNode accent>S3</DbNode>
                <MiniArrow />
                <NodeChip>partitions</NodeChip>
              </Diagram>
            ),
            children:
              "Paginates a bucket, assigns each object a dense position, and supports versioned-prefix snapshots or monotonic-key follow mode.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Local source",
            diagram: (
              <Diagram>
                <NodeChip accent>disk</NodeChip>
                <MiniArrow />
                <NodeChip>partitions</NodeChip>
              </Diagram>
            ),
            children:
              "Walks a local directory tree; each .jsonl.gz file becomes one partition. Used for development and dry-run replay.",
          },
          {
            accent: "purple",
            badge: 3,
            title: "Dynamo format decoder",
            diagram: (
              <Diagram>
                <NodeChip>.gz</NodeChip>
                <MiniArrow />
                <NodeChip accent>units</NodeChip>
              </Diagram>
            ),
            children:
              'Decompresses gzip, parses JSONL line-by-line, validates against schema "dynamo.request.trace.v1", and emits typed DatasetActionV1 request units.',
          },
          {
            accent: "green",
            badge: 4,
            title: "Immutable identity",
            diagram: (
              <Diagram>
                <NodeChip>bucket+key+gen</NodeChip>
                <MiniArrow />
                <NodeChip accent>BLAKE3</NodeChip>
              </Diagram>
            ),
            children:
              "Partition identity is BLAKE3(bucket, key, generation-token, size). Known at listing time — no re-read needed. Content digest is verified separately on acquisition.",
          },
          {
            accent: "orange",
            badge: 5,
            title: "Local checkpoint backend",
            diagram: (
              <Diagram>
                <NodeChip accent>cursor</NodeChip>
                <MiniArrow />
                <DbNode>disk</DbNode>
              </Diagram>
            ),
            children:
              "Persists per-partition read position as an atomic fsync'd file. On restart the source skips fully-processed partitions and resumes mid-partition from the last committed offset.",
          },
          {
            accent: "red",
            badge: 6,
            title: "None / CAS backends",
            diagram: (
              <Diagram>
                <NodeChip>5F1</NodeChip>
                <MiniArrow />
                <NodeChip accent>5F2</NodeChip>
              </Diagram>
            ),
            children:
              'None backend: no persistence, every run starts fresh. Conditional object-store (CAS) backend: compare-and-swap write to S3/MinIO for distributed checkpoint coordination.',
          },
          {
            accent: "yellow",
            badge: 7,
            title: "Source retry and backoff",
            diagram: (
              <Diagram>
                <NodeChip>err</NodeChip>
                <MiniArrow />
                <NodeChip accent>backoff</NodeChip>
              </Diagram>
            ),
            children:
              "Clock-driven linear backoff with a configured ceiling. Authorization failures invalidate the shared credential before waiting. Each partition has its own retry counter toward a durable-hole threshold.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "S3 source", path: "rust/runtime/src/streaming/sources/s3.rs" },
          { label: "Local source", path: "rust/runtime/src/streaming/sources/local.rs" },
          { label: "Dynamo format", path: "rust/runtime/src/streaming/formats/streaming_dynamo.rs" },
          { label: "Local checkpoint", path: "rust/runtime/src/streaming/checkpoints/local.rs" },
          { label: "None backend", path: "rust/runtime/src/streaming/checkpoints/none.rs" },
        ]}
      />
    </div>
  );
}
