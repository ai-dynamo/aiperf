/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { HubSpoke, Diagram, NodeChip, DbNode, MiniArrow } from "../../chalk/index.js";
import { EvidenceRow, PageIntro } from "./shared.js";

// Systems Chalk hub-and-spoke of the EndpointsView: dialects own payload/response semantics, and
// each worker builds a dense prepared table so per-turn dispatch skips registry/config work.

/** EndpointsView: dialects own payload/response semantics; workers build dense prepared tables. */
export function EndpointsPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Dialect preparation">
        Endpoint dialects own payload and response semantics. Validation resolves authored profiles once; each worker
        builds a dense prepared table so request dispatch avoids repeated registry and configuration work.
      </PageIntro>

      <HubSpoke
        hub={{
          kicker: "AIPERF · ENDPOINTS",
          title: "How are dialects prepared?",
          body: "Profiles validated once, tables built per worker, dialects own the wire.",
        }}
        spokes={[
          {
            accent: "blue",
            badge: 1,
            title: "Resolve profiles",
            diagram: (
              <Diagram>
                <NodeChip>profiles</NodeChip>
                <MiniArrow />
                <NodeChip accent>EndpointRegistry</NodeChip>
              </Diagram>
            ),
            children: "Endpoint profiles (id + model + raw config) look up a factory by EndpointId in the registry.",
          },
          {
            accent: "cyan",
            badge: 2,
            title: "Strict validation",
            diagram: (
              <Diagram>
                <NodeChip>raw</NodeChip>
                <MiniArrow />
                <NodeChip accent>EndpointKey</NodeChip>
              </Diagram>
            ),
            children: "Strict validation lowers raw to effective config, yielding stable dense profile identities.",
          },
          {
            accent: "green",
            badge: 3,
            title: "Prepare worker table",
            diagram: (
              <Diagram>
                <NodeChip>prepare_worker()</NodeChip>
                <MiniArrow />
                <DbNode accent>PreparedEndpointTable</DbNode>
              </Diagram>
            ),
            children:
              "The PreparedEndpointTableFactory blueprint builds a worker-local tokenizer and a dense-lookup table.",
          },
          {
            accent: "yellow",
            badge: 4,
            title: "Format per turn",
            diagram: (
              <Diagram>
                <NodeChip>PreparedTurn</NodeChip>
                <MiniArrow />
                <NodeChip accent>dialect</NodeChip>
              </Diagram>
            ),
            children: "Per turn the endpoint dialect formats the payload, headers, and response parser from content + tokens.",
          },
          {
            accent: "purple",
            badge: 5,
            title: "Bind transport",
            diagram: (
              <Diagram>
                <NodeChip>HTTP URI</NodeChip>
                <MiniArrow />
                <NodeChip accent>gRPC tensors</NodeChip>
              </Diagram>
            ),
            children: "HTTP and gRPC share endpoint identity but bind different wire representations.",
          },
          {
            accent: "red",
            badge: 6,
            title: "Emit observations",
            diagram: (
              <Diagram>
                <NodeChip accent>parser</NodeChip>
                <MiniArrow />
                <NodeChip>tokens · usage</NodeChip>
              </Diagram>
            ),
            children: "Parsers reconcile provider usage and token classification into tokens, usage, and endpoint metrics.",
          },
          {
            accent: "orange",
            badge: 7,
            title: "Dialect families",
            diagram: (
              <Diagram>
                <NodeChip>OpenAI + Anthropic</NodeChip>
                <MiniArrow />
                <NodeChip accent>KServe · Riva · vLLM</NodeChip>
              </Diagram>
            ),
            children: "Families span OpenAI + Anthropic, KServe HTTP/gRPC, NVIDIA Riva, and vLLM + specialized endpoints.",
          },
        ]}
      />

      <EvidenceRow
        items={[
          { label: "Endpoint trait", path: "rust/aiperf/src/endpoints/endpoints.rs" },
          { label: "Endpoint registry", path: "rust/aiperf/src/endpoints/registry.rs" },
          { label: "HTTP preparation", path: "rust/aiperf/src/runner_protocol/turn_execution.rs" },
          { label: "gRPC binding", path: "rust/aiperf/src/transport_grpc/binding.rs" },
        ]}
      />
    </div>
  );
}
