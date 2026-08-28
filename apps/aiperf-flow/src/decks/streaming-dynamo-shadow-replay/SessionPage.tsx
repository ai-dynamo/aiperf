/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, MiniArrow, RoundNode } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

//! Session page: ConversationCoordinator, SessionClosurePolicy, ActionHost, TurnClosureIntake.

/** Session layer: join fragments, decide closure, emit turn actions. */
export function SessionPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Join fragments into conversations">
        A single Dynamo trace file may interleave records from many concurrent sessions. The conversation coordinator
        groups them by session key, folds observed endpoint replies into the same durable transcript as authored turns,
        and re-emits an in-flight turn under its identical stable action identity after a restart. Once a session is
        complete, the closure policy retires it; the action host drains its pending turns into the pipeline via the
        turn-closure intake queue.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "STREAMING · SESSION LAYER",
          title: "How are interleaved records turned into ordered conversations?",
          body: "Fragment join → closure decision → action emission. Sessions survive restarts via stable action identity.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Fragment join",
            diagram: (
              <Diagram>
                <RoundNode>A</RoundNode>
                <RoundNode>B</RoundNode>
                <MiniArrow />
                <NodeChip accent>session</NodeChip>
              </Diagram>
            ),
            children:
              "Records from the same session key are joined across source partitions and chunk boundaries into one durable in-memory transcript.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Cross-partition join",
            diagram: (
              <Diagram>
                <NodeChip>file 1</NodeChip>
                <MiniArrow />
                <NodeChip accent>file 2</NodeChip>
              </Diagram>
            ),
            children:
              "A session may span multiple files. The coordinator holds open sessions until the source signals completeness or a finite seal closes the stream.",
          },
          {
            accent: "purple",
            badge: 3,
            title: "Stable action identity",
            diagram: (
              <Diagram>
                <NodeChip>restart</NodeChip>
                <MiniArrow />
                <NodeChip accent>same id</NodeChip>
              </Diagram>
            ),
            children:
              "Each turn's action id is derived from causal and semantic inputs only — not from wall-clock time. The pipeline can re-emit it after a restart without duplicating delivered results.",
          },
          {
            accent: "green",
            badge: 4,
            title: "Session closure policy (P1B)",
            diagram: (
              <Diagram>
                <NodeChip>seal</NodeChip>
                <MiniArrow />
                <NodeChip accent>close</NodeChip>
              </Diagram>
            ),
            children:
              "A finite seal with no causal gap closes the session immediately. A causal gap with finite_seal_requires_complete set fails the session instead of silently dropping turns.",
          },
          {
            accent: "orange",
            badge: 5,
            title: "Action host (plan-P2)",
            diagram: (
              <Diagram>
                <NodeChip accent>host</NodeChip>
                <MiniArrow />
                <NodeChip>Request</NodeChip>
              </Diagram>
            ),
            children:
              "Multiplexed state-only sink. Holds per-session turn state locally; emits a Request action for each completed turn without blocking the session loop.",
          },
          {
            accent: "red",
            badge: 6,
            title: "Turn closure intake (P2)",
            diagram: (
              <Diagram>
                <NodeChip>turn</NodeChip>
                <MiniArrow />
                <NodeChip accent>VecDeque</NodeChip>
              </Diagram>
            ),
            children:
              "Rc<RefCell<VecDeque>> — zero-copy, worker-local closed-turn queue. The pipeline drains it in its fused event loop without any cross-thread synchronization.",
          },
          {
            accent: "yellow",
            badge: 7,
            title: "Quarantine on failure",
            diagram: (
              <Diagram>
                <NodeChip>gap</NodeChip>
                <MiniArrow />
                <NodeChip accent>quarantine</NodeChip>
              </Diagram>
            ),
            children:
              "Sessions with unresolvable causal gaps or repeated decode errors are quarantined rather than silently dropped, and are counted toward the authored quarantine admission fence.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "Conversation coordinator", path: "rust/runtime/src/streaming/session/conversation.rs" },
          { label: "Closure policy", path: "rust/runtime/src/streaming/session/closure.rs" },
          { label: "Action host", path: "rust/runtime/src/streaming/session/host.rs" },
          { label: "Closure seam", path: "rust/runtime/src/streaming/closure.rs" },
          { label: "Stable action id", path: "rust/runtime/src/streaming/identity.rs" },
        ]}
      />
    </div>
  );
}
