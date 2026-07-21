/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { bandHeader, card, chip, dashed, DeckDiagram, EvidenceRow, flow, panel, PageIntro } from "./shared.js";

// Ported from the EndpointsView page: dialect preparation.

const nodes: Node[] = [
  bandHeader("b-startup", "Startup validation", 0, 0),
  panel("profiles", "endpoint profiles", "id + model + raw config", 0, 60),
  card("registry", "EndpointRegistry", undefined, "factory lookup by EndpointId", 280, 60),
  panel("strict", "strict validation", "raw → effective config", 580, 60),
  card("identities", "profile identities", undefined, "stable dense EndpointKey", 840, 60),

  bandHeader("b-worker", "Worker preparation", 0, 200),
  panel("factory", "PreparedEndpointTableFactory", "shared startup blueprint", 0, 260),
  card("prepare", "prepare_worker()", undefined, "worker-local tokenizer + bindings", 340, 260),
  card("table", "PreparedEndpointTable", undefined, "dense lookup by EndpointKey", 660, 260),

  bandHeader("b-perturn", "Per-turn request and response", 0, 400),
  panel("turn", "PreparedTurn", "content + token counts", 0, 460),
  card("dialect", "Endpoint dialect", undefined, "format payload · headers · parser", 280, 460),
  card("binding", "transport binding", undefined, "HTTP URI/body or gRPC tensors", 580, 460),
  card("observations", "observations", undefined, "tokens · usage · endpoint metrics", 860, 460),

  bandHeader("b-families", "Dialect families", 0, 600),
  chip("f-openai", "OpenAI + Anthropic", 0, 660),
  chip("f-kserve", "KServe HTTP/gRPC", 220, 660),
  chip("f-riva", "NVIDIA Riva", 420, 660),
  chip("f-vllm", "vLLM + specialized", 600, 660),
];

const edges: Edge[] = [
  flow("profiles", "registry"),
  flow("registry", "strict"),
  flow("strict", "identities"),
  flow("factory", "prepare"),
  flow("prepare", "table"),
  flow("turn", "dialect"),
  flow("dialect", "binding"),
  flow("binding", "observations"),
  dashed("observations", "dialect"),
];

/** EndpointsView: dialects own payload/response semantics; workers build dense prepared tables. */
export function EndpointsPage(): React.JSX.Element {
  return (
    <div className="flex h-full w-full flex-col gap-4">
      <PageIntro title="Dialect preparation">
        Endpoint dialects own payload and response semantics. Validation resolves authored profiles once; each worker
        builds a dense prepared table so request dispatch avoids repeated registry and configuration work.
      </PageIntro>

      <DeckDiagram nodes={nodes} edges={edges} height={600} />

      <Grid columns={3} gap={16}>
        <Callout tone="info" title="Open registry">
          New dialects register factories; the core transport does not gain an endpoint-type switch.
        </Callout>
        <Callout tone="info" title="Transport-native binding">
          HTTP and gRPC share endpoint identity but prepare different wire representations.
        </Callout>
        <Callout tone="success" title="Usage authority">
          Endpoint parsers reconcile provider usage and token classification before emitting observer facts.
        </Callout>
      </Grid>

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
