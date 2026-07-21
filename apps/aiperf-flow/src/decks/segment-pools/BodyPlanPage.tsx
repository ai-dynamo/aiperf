/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ported from `docs/canvases/segment-pools-and-body-plans.canvas.tsx` `PageBodyPlan`
//! (rust/aiperf/src/body_plan.rs, `JsonBodyMaterializer`). Three toggles — `rawMode`,
//! `stream`, `tools` — recompute a token list that is rendered as the "materialized
//! bytes" a `JsonBodyMaterializer` would emit: `Segment`/`Segments` fields are wire
//! clones spliced straight through, `Literal` fields are the only bytes actually
//! serialized on the hot path, and `stream`/`max_tokens` are an override tail merged
//! in afterward.

import { useState } from "react";
import clsx from "clsx";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Callout } from "../../prose/Callout.js";
import { Table } from "../../prose/Table.js";
import { Legend } from "../../prose/Legend.js";
import { Stat } from "../../prose/Stat.js";
import { inkClassName, surfaceClassName, strokeClassName, categoryBgTintClassName } from "../../theme/tokens.js";
import type { CategoryRole } from "../../theme/tokens.js";

const MSG_WIRE_1 = `{"role":"user","content":"What is 2+2?"}`;
const MSG_WIRE_2 = `{"role":"assistant","content":"4"}`;
const TOOLS_WIRE = `[{"type":"function","function":{"name":"calc"}}]`;
const RAW_WIRE = `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":false}`;

type TokKind = "lit" | "seg" | "tail" | "punc";
type Tok = { text: string; kind: TokKind };

const KIND_COLOR: Record<TokKind, CategoryRole | "tertiary"> = {
  seg: "green", // spliced wire (clone bytes)
  lit: "blue", // literal (serde_json)
  tail: "yellow", // override tail
  punc: "tertiary",
};

function buildTokens(rawMode: boolean, stream: boolean, tools: boolean): Tok[] {
  if (rawMode) {
    return [
      { text: RAW_WIRE, kind: "seg" },
      { text: `,"stream":${stream}`, kind: "tail" },
    ];
  }
  const toks: Tok[] = [
    { text: `{`, kind: "punc" },
    { text: `"messages":[`, kind: "punc" },
    { text: MSG_WIRE_1, kind: "seg" },
    { text: `,`, kind: "punc" },
    { text: MSG_WIRE_2, kind: "seg" },
    { text: `]`, kind: "punc" },
    { text: `,"model":"gpt-4"`, kind: "lit" },
  ];
  if (tools) {
    toks.push({ text: `,"tools":${TOOLS_WIRE}`, kind: "seg" });
  }
  toks.push({ text: `,"stream":${stream}`, kind: "tail" });
  toks.push({ text: `,"max_tokens":128`, kind: "tail" });
  toks.push({ text: `}`, kind: "punc" });
  return toks;
}

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={clsx(
        "rounded-none border px-3 py-1 text-xs font-semibold transition-colors",
        strokeClassName("secondary"),
        active
          ? clsx(categoryBgTintClassName("blue"), inkClassName("primary"))
          : clsx(surfaceClassName("elevated"), inkClassName("tertiary")),
      )}
    >
      {children}
    </button>
  );
}

/**
 * Interactive `BodyPlan` materializer: three toggles (`BodyPlan::Fields` vs
 * `BodyPlan::Raw`, `stream`, `tools`) recompute a token list rendered as the
 * "materialized bytes" a `JsonBodyMaterializer` would splice together, along with
 * the field-level plan and hot-path serde_json op count for the current combination.
 */
export function BodyPlanPage(): React.JSX.Element {
  const [rawMode, setRawMode] = useState(false);
  const [stream, setStream] = useState(true);
  const [tools, setTools] = useState(false);

  const toks = buildTokens(rawMode, stream, tools);
  const serdeOps = rawMode ? 0 : toks.filter((tok) => tok.kind === "lit").length;
  const splices = toks.filter((tok) => tok.kind === "seg").length;

  const planRows = rawMode
    ? [
        { field: "Raw(H7)", kind: "Segment wire", note: "complete body — endpoint bypassed", tone: "success" as const },
        { field: "+ overrides", kind: "tail splice", note: "stream / model patched in", tone: "warning" as const },
      ]
    : [
        { field: "messages", kind: "FieldValue::Segments", note: "[H2, H3] → wire clones", tone: "success" as const },
        { field: "model", kind: "FieldValue::Literal", note: "serde_json::to_writer", tone: "neutral" as const },
        ...(tools
          ? [{ field: "tools", kind: "FieldValue::Segment", note: "H5 → raw wire clone", tone: "success" as const }]
          : []),
        { field: "stream", kind: "override tail", note: "merge_overrides", tone: "warning" as const },
        { field: "max_tokens", kind: "override tail", note: "merge_overrides", tone: "warning" as const },
      ];

  return (
    <Stack gap={16}>
      <div>
        <h2 className={clsx("text-xl font-bold", inkClassName("primary"))}>
          BodyPlan — shape now, bytes later
        </h2>
        <p className={clsx("mt-2 text-sm", inkClassName("secondary"))}>
          A <code className="font-mono">BodyPlan</code> declares which fields exist and which slots
          are filled by segment handles vs literals. The{" "}
          <code className="font-mono">JsonBodyMaterializer</code> walks it once and produces{" "}
          <code className="font-mono">Bytes</code>: literals are the{" "}
          <span className="font-semibold">only</span> thing serialized on the hot path — segment
          fields are pre-serialized wires cloned straight through.
        </p>
      </div>

      <Row gap={8} wrap align="center">
        <ToggleButton active={!rawMode} onClick={() => setRawMode(false)}>
          BodyPlan::Fields
        </ToggleButton>
        <ToggleButton active={rawMode} onClick={() => setRawMode(true)}>
          BodyPlan::Raw
        </ToggleButton>
        <div className="flex-1" />
        {!rawMode && (
          <ToggleButton active={tools} onClick={() => setTools((v) => !v)}>
            tools {tools ? "on" : "off"}
          </ToggleButton>
        )}
        <ToggleButton active={stream} onClick={() => setStream((v) => !v)}>
          stream {stream ? "on" : "off"}
        </ToggleButton>
      </Row>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className={clsx("rounded-none border", strokeClassName("secondary"))}>
          <div
            className={clsx(
              "border-b px-3 py-2 text-xs font-semibold uppercase tracking-wide",
              strokeClassName("secondary"),
              inkClassName("secondary"),
            )}
          >
            Plan — {rawMode ? "Raw(Handle)" : "Fields[..]"}
          </div>
          <div className="p-3">
            <Table
              columns={[
                { key: "field", label: "Field" },
                { key: "kind", label: "Kind" },
                { key: "note", label: "Note" },
              ]}
              rows={planRows.map((row) => ({ ...row }))}
            />
          </div>
        </div>

        <div className={clsx("rounded-none border", strokeClassName("secondary"))}>
          <div
            className={clsx(
              "border-b px-3 py-2 text-xs font-semibold uppercase tracking-wide",
              strokeClassName("secondary"),
              inkClassName("secondary"),
            )}
          >
            Materialized bytes — MaterializedRequest.body
          </div>
          <div className="p-3">
            <pre
              className={clsx(
                "whitespace-pre-wrap break-all rounded-none border p-2.5 font-mono text-[11.5px] leading-[19px]",
                surfaceClassName("panel"),
                strokeClassName("secondary"),
              )}
            >
              {toks.map((tok, i) => {
                const color = KIND_COLOR[tok.kind];
                return (
                  <code
                    key={i}
                    className={clsx(
                      "rounded-none",
                      tok.kind !== "punc" && "px-0.5 py-px",
                      color !== "tertiary" ? `text-category-${color}` : inkClassName("tertiary"),
                      tok.kind === "seg" && categoryBgTintClassName("green"),
                      tok.kind === "tail" && categoryBgTintClassName("yellow"),
                    )}
                  >
                    {tok.text}
                  </code>
                );
              })}
            </pre>
            <div className="mt-3">
              <Legend
                entries={[
                  { color: "green", label: "segment wire — cloned" },
                  { color: "blue", label: "literal — serialized" },
                  { color: "yellow", label: "override tail" },
                ]}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Stat
          value={serdeOps}
          label="serde_json ops (hot path)"
          tone={serdeOps === 0 ? "positive" : "neutral"}
        />
        <Stat value={splices} label="byte splices (zero re-serialize)" tone="positive" />
        <Stat
          value={rawMode ? "bypass" : "compose"}
          label={rawMode ? "endpoint bypassed" : "endpoint.format_payload"}
        />
      </div>

      <Callout tone="info" title="The types (body_plan.rs)">
        <pre className="mt-1 overflow-x-auto whitespace-pre font-mono text-xs leading-[18px]">
{`pub enum FieldValue {
    Literal(Value),                    // serialized on the hot path
    Segment(Handle),                   // one wire cloned from the store
    Segments(SmallVec<[Handle; 1]>),   // [ wire, wire, ... ] joined
    Wires(SmallVec<[Bytes; 1]>),       // dynamic content, no store lookup
}

pub enum BodyPlan {
    Raw(Handle),                                   // whole body passthrough
    Fields(SmallVec<[(FieldName, FieldValue); 8]>),
}`}
        </pre>
      </Callout>
    </Stack>
  );
}
