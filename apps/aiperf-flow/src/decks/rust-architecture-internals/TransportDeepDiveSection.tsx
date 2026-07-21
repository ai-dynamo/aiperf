/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Section 05 — cold endpoint preparation feeds a hot wire lane. Cold path (registry →
//! effective config → prepared table → EndpointKey) versus the HTTP or gRPC hot path down to
//! the worker measurement wrapper and terminal order. Ported from `TransportDeepDive`.

import { useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { inkClassName } from "../../theme/tokens.js";
import {
  Segmented,
  SectionHeading,
  SourcesRow,
  SectionShell,
  FlowFrame,
  headerNode,
  cardNode,
  panelNode,
  flowEdge,
  rank,
  type Detail,
} from "./parts.js";

type Wire = "http" | "grpc";

function buildNodes(wire: Wire): Node[] {
  const nodes: Node[] = [
    headerNode("band-cold", 0, 0, "Cold path · once per worker"),
    cardNode("registry-prepare", 20, 50, "EndpointRegistry::prepare", "resolve + validate descriptor"),
    cardNode("effective", 20, 150, "EffectiveEndpointConfig", "validated immutable snapshot"),
    cardNode("prepared-table", 20, 250, "PreparedEndpointTable", "worker-local dense vector"),
    cardNode("endpoint-key", 20, 350, "EndpointKey(u32)", "direct index lookup", undefined, "primary"),

    headerNode("band-hot", 360, 0, `Hot path · ${wire === "http" ? "HTTP" : "gRPC"}`),
    cardNode(
      "sink",
      360,
      50,
      wire === "http" ? "TransportSink" : "GrpcTransportSink",
      "prepared binding lookup",
      wire === "grpc" ? "endpoint-aware required" : "prepared dialect path",
      "primary",
    ),
    cardNode("reduce", 360, 280, "reduce_parsed_response", "tokens + usage + content"),
    panelNode("measure", 360, 380, "WORKER MEASUREMENT WRAPPER", "register metadata → on_arrival → await dispatch → record_response", "primary"),
  ];

  if (wire === "http") {
    nodes.push(
      cardNode("hot-1", 660, 50, "Hyper response body", "byte chunks + chunk-arrival Clock timestamp", "delimiter scan across split boundaries"),
      cardNode("hot-2", 660, 170, "SSE reader", "parse complete events · keep reading after [DONE]"),
      cardNode("hot-3", 660, 280, "endpoint parser", "ParsedResponse + carried content"),
    );
  } else {
    nodes.push(
      cardNode("hot-1", 660, 50, "prepared gRPC binding", "unary | server stream | bidi", "selected from binding + request.streaming"),
      cardNode("hot-2", 660, 170, "Tonic dispatch", "message stream + send anchor"),
      cardNode("hot-3", 660, 280, "endpoint parser", "ParsedResponse + carried content"),
    );
  }
  return nodes;
}

const coldEdges: Edge[] = [
  flowEdge("e-prep-eff", "registry-prepare", "effective"),
  flowEdge("e-eff-table", "effective", "prepared-table"),
  flowEdge("e-table-key", "prepared-table", "endpoint-key"),
];

function buildEdges(): Edge[] {
  return [
    ...coldEdges,
    flowEdge("e-key-sink", "endpoint-key", "sink"),
    flowEdge("e-sink-hot1", "sink", "hot-1"),
    flowEdge("e-hot1-hot2", "hot-1", "hot-2"),
    flowEdge("e-hot2-hot3", "hot-2", "hot-3"),
    flowEdge("e-hot3-reduce", "hot-3", "reduce"),
    flowEdge("e-reduce-measure", "reduce", "measure"),
  ];
}

/** Section 05 diagram: cold endpoint preparation and the wire-selected hot lane. */
export function TransportDeepDiveSection({ detail }: { detail: Detail }): React.JSX.Element {
  const [wire, setWire] = useState<Wire>("http");
  return (
    <SectionShell>
      <Row gap={16} align="end" justify="space-between" wrap>
        <SectionHeading
          number="05"
          title="Cold endpoint preparation feeds a hot wire lane"
          subtitle="Authored endpoint IDs resolve into immutable prepared bindings and dense keys before HTTP or gRPC dispatch begins."
        />
        <Segmented
          ariaLabel="Wire"
          value={wire}
          onChange={setWire}
          options={[
            { id: "http", label: "HTTP / SSE" },
            { id: "grpc", label: "gRPC" },
          ]}
        />
      </Row>

      <FlowFrame nodes={buildNodes(wire)} edges={buildEdges()} height={540} />
      <p className={`text-center text-xs ${inkClassName("secondary")}`}>
        TERMINAL ORDER: token callbacks* → on_usage → on_endpoint_metrics? → on_terminal → record_response
      </p>

      <Grid columns={2} gap={14}>
        <Callout tone="info" title="Send-anchored cancellation">
          HTTP and gRPC compute cancellation from the recorded send-completion timestamp.
        </Callout>
        <Callout tone="warning" title="HTTP dispatch branches">
          The prepared dialect branch uses shared response reduction; the inline chat branch parses chunks inside{" "}
          <Code inline>TransportSink</Code>.
        </Callout>
      </Grid>
      {rank(detail) > 0 && (
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          * Output may arrive through <Code inline>on_output_tokens</Code> or <Code inline>on_classified_token</Code>;{" "}
          <Code inline>measure_dispatch</Code> emits <Code inline>on_arrival</Code> before transport dispatch.
        </p>
      )}
      <SourcesRow
        detail={detail}
        paths={[
          { label: "endpoint registry", path: "rust/runtime/src/endpoints/registry.rs" },
          { label: "HTTP dispatch", path: "rust/runtime/src/transport/http/sink/endpoint_dispatch.rs" },
          { label: "SSE reader", path: "rust/runtime/src/transport/http/sse/reader.rs" },
          { label: "gRPC sink", path: "rust/runtime/src/transport/grpc/sink.rs" },
          { label: "shared reduce", path: "rust/runtime/src/transport/reduce.rs" },
        ]}
      />
    </SectionShell>
  );
}
