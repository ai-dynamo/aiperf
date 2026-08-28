/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, MiniArrow, DbNode } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

//! Shadow Replay page: ScheduledRequestSink, ActionInventory, shadow_replay workload registration.

/** Shadow Replay workload: re-execute recorded requests. */
export function ShadowReplayPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Re-execute recorded requests against a live endpoint">
        The shadow-replay workload type is registered in the engine as a first-class counterpart to{" "}
        <code>scheduled</code> and <code>graph</code>. Config-v2's <code>shadow_replay:</code> section selects the
        stream, action bindings, timing interpretation, ordering policy, overload behavior, and checkpoint cadence.
        The scheduled-request action sink issues each recorded request against a real HTTP/gRPC endpoint —
        preserving the original inter-request timing offsets — and the action inventory ledger tracks in-flight,
        completed, and failed actions for dense gap-closure before result finalization.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "STREAMING · SHADOW REPLAY WORKLOAD",
          title: "How are recorded Dynamo requests re-issued against a real endpoint?",
          body: "Config-v2 shadow_replay: section → registered workload → scheduled-request sink → action inventory → results.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "shadow_replay workload type",
            diagram: (
              <Diagram>
                <NodeChip>Config v2</NodeChip>
                <MiniArrow />
                <NodeChip accent>shadow_replay</NodeChip>
              </Diagram>
            ),
            children:
              "Registered in AIPerfRegistry via BuiltinStreamingExtension (A-REG). Detected by workload_kind() from the presence of a shadow_replay: section in the config — distinct from graph and scheduled.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Config-v2 section",
            diagram: (
              <Diagram>
                <NodeChip>stream:</NodeChip>
                <MiniArrow />
                <NodeChip accent>actions:</NodeChip>
              </Diagram>
            ),
            children:
              'shadow_replay: names the dataset_streams: entry to use, maps action kinds (request / graph_node / session_terminal) to sink implementations, and configures timing mode (relative / absolute), ordering, overload, and checkpoint.',
          },
          {
            accent: "purple",
            badge: 3,
            title: "Timing interpretation",
            diagram: (
              <Diagram>
                <NodeChip>recorded ms</NodeChip>
                <MiniArrow />
                <NodeChip accent>offset</NodeChip>
              </Diagram>
            ),
            children:
              'Relative mode: timestamps are offsets from the replay origin — the first action fires immediately and later actions fire at their recorded inter-arrival intervals. Absolute mode: event times are UTC instants from the original trace.',
          },
          {
            accent: "green",
            badge: 4,
            title: "Scheduled-request sink (P4)",
            diagram: (
              <Diagram>
                <NodeChip accent>action</NodeChip>
                <MiniArrow />
                <DbNode>HTTP · gRPC</DbNode>
              </Diagram>
            ),
            children:
              "Issues one real request per action against the configured endpoint (chat / completions / responses over HTTP or gRPC). Respects original inter-request timing. endpoint_retry_safety is Unproven — a non-zero retry limit is refused because a retry duplicates measured load.",
          },
          {
            accent: "orange",
            badge: 5,
            title: "Action inventory ledger (P4)",
            diagram: (
              <Diagram>
                <NodeChip>in-flight</NodeChip>
                <MiniArrow />
                <NodeChip accent>gap-close</NodeChip>
              </Diagram>
            ),
            children:
              "Dense gap-closure ledger: tracks which action positions are in-flight, successfully completed, or permanently failed. Advancement of the delivery frontier requires no holes — a failed action is a durable hole, not a retry trigger.",
          },
          {
            accent: "red",
            badge: 6,
            title: "Overload policy",
            diagram: (
              <Diagram>
                <NodeChip>behind</NodeChip>
                <MiniArrow />
                <NodeChip accent>backpressure</NodeChip>
              </Diagram>
            ),
            children:
              "Backpressure mode stalls source acquisition when the action sink cannot keep up. Shed mode explicitly drops admitted work with lossy semantics and increments the shed counter. Both are authored in shadow_replay.overload.mode.",
          },
          {
            accent: "yellow",
            badge: 7,
            title: "Config-v2 preflight (V1)",
            diagram: (
              <Diagram>
                <NodeChip>validate</NodeChip>
                <MiniArrow />
                <NodeChip accent>preflight</NodeChip>
              </Diagram>
            ),
            children:
              "V1 added streaming-specific Config-v2 validation: StreamingReliabilityPolicy round-trips through strict serde, dataset_streams items are checked for duplicate ids, and the shadow_replay stream reference is resolved before any S3 listing starts.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "shadow_replay workload", path: "rust/runtime/src/engine/streaming_execution.rs" },
          { label: "Workload kind", path: "rust/runtime/src/config/model/workload_kind.rs" },
          { label: "Config model", path: "rust/runtime/src/config/model/dataset_stream.rs" },
          { label: "Scheduled-request sink", path: "rust/runtime/src/streaming/action/scheduled_request.rs" },
          { label: "Action inventory", path: "rust/runtime/src/streaming/action/host/inventory.rs" },
          { label: "Registry wiring", path: "rust/runtime/src/engine/registry.rs" },
          { label: "Config validation", path: "rust/runtime/src/config/validate.rs" },
        ]}
      />
    </div>
  );
}
