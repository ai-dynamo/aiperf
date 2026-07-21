/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Source-grounded evidence panel, ported from the canvas's `EngineerInspector`: selecting any
//! atlas node or edge reveals its status, symbol, source path, and proof/boundary string verbatim.

import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Code } from "../../prose/Code.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { inkClassName } from "../../theme/tokens.js";
import type { CategoryRole } from "../../theme/tokens.js";
import {
  NODE_BY_ID,
  EDGE_BY_ID,
  NODES,
  statusLabel,
  statusTone,
  type Status,
} from "./data.js";

const STATUS_TONE: Record<ReturnType<typeof statusTone>, CategoryRole | undefined> = {
  success: "green",
  warning: "yellow",
  neutral: undefined,
};

function StatusPill({ status }: { status: Status }): React.JSX.Element {
  return <Eyebrow tone={STATUS_TONE[statusTone(status)]}>{statusLabel(status)}</Eyebrow>;
}

/** Engineer inspector for the currently selected node/edge id. Falls back to the controller node. */
export function Inspector({ selectedId }: { selectedId: string }): React.JSX.Element {
  const edge = EDGE_BY_ID.get(selectedId);
  if (edge) {
    const from = NODE_BY_ID.get(edge.from);
    const to = NODE_BY_ID.get(edge.to);
    return (
      <div className="rounded-lg border border-stroke-secondary px-4 py-3 shadow-sm">
        <Row justify="space-between" align="center">
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Selected flow</span>
          <StatusPill status={edge.status} />
        </Row>
        <Stack gap={8} className="mt-2">
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
            {from?.label} → {to?.label}
          </span>
          <span className={`text-sm ${inkClassName("secondary")}`}>{edge.payload}</span>
          <Code inline>{edge.id}</Code>
        </Stack>
      </div>
    );
  }

  const node = NODE_BY_ID.get(selectedId) ?? NODES[2];
  return (
    <div className="rounded-lg border border-stroke-secondary px-4 py-3 shadow-sm">
      <Row justify="space-between" align="center">
        <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Engineer inspector</span>
        <StatusPill status={node.status} />
      </Row>
      <Stack gap={10} className="mt-2">
        <div>
          <div className={`text-sm font-semibold ${inkClassName("primary")}`}>{node.label}</div>
          <div className={`mt-0.5 text-sm ${inkClassName("secondary")}`}>{node.detail}</div>
        </div>
        <Stack gap={3}>
          <Eyebrow>Symbol</Eyebrow>
          <Code>{node.symbol}</Code>
        </Stack>
        <Stack gap={3}>
          <Eyebrow>Source</Eyebrow>
          <span className={`break-all text-sm ${inkClassName("secondary")}`}>{node.path}</span>
        </Stack>
        <Stack gap={3}>
          <Eyebrow>Proof / boundary</Eyebrow>
          <span className={`text-sm ${inkClassName("secondary")}`}>{node.proof}</span>
        </Stack>
      </Stack>
    </div>
  );
}
