/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import clsx from "clsx";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Callout } from "../../prose/Callout.js";
import { Swatch } from "../../prose/Swatch.js";
import {
  surfaceClassName,
  strokeClassName,
  inkClassName,
  categoryClassName,
  categoryBgTintClassName,
  type CategoryRole,
} from "../../theme/tokens.js";

//! Payloads page — the six disjoint blake3 hash domains a `Segment::Payload` can carry.
//!
//! Ported from `docs/canvases/segment-pools-and-body-plans.canvas.tsx`, `PagePayloads`
//! (the `sel` domain selector around line 560) and its `DOMAINS` table. Every recipe is
//! also framed by the parent segment id and the shared version constant, which is shown
//! as the leading two rows of every recipe regardless of the selected domain.

const HASH_VERSION = `b"aiperf-dataset-segment-v1\\0"`;
const PARENT_ID_ROW = `parent SegmentId + "\\0"`;

type Domain = {
  key: string;
  name: string;
  color: CategoryRole;
  fields: string;
  use: string;
  prefix: string;
  recipe: string[];
};

// The canvas source uses a "pink" category for Media; this app's `CategoryRole` has no
// pink hue, so Media is mapped onto "red" (unused by the other five domains) to keep all
// six domains visually distinct.
const DOMAINS: Domain[] = [
  {
    key: "message",
    name: "Message",
    color: "blue",
    fields: "role, wire: Bytes, tokens: Box<[u32]>",
    use: "Pre-serialized endpoint message; wire spliced into messages[]",
    prefix: `"message\\0"`,
    recipe: ["role.as_str()", `"\\0"`, "each token (u32 LE)", `"\\0"`, "full wire bytes"],
  },
  {
    key: "text",
    name: "Text",
    color: "cyan",
    fields: "role, bytes: Bytes, token_count: u32",
    use: "Text-only field; token ids folded into id, not stored",
    prefix: `"text-only\\0"`,
    recipe: ["role.as_str()", `"\\0"`, "token ids (hashed at intern time)"],
  },
  {
    key: "raw",
    name: "Raw",
    color: "purple",
    fields: "wire: Bytes",
    use: "Complete JSON body / tools / headers → BodyPlan::Raw or a field",
    prefix: `"raw\\0"`,
    recipe: ["wire bytes"],
  },
  {
    key: "tokenids",
    name: "TokenIds",
    color: "green",
    fields: "token_ids: Box<[u32]>",
    use: "Token-native path; gRPC / validation, not spliced into JSON",
    prefix: `"token-ids\\0"`,
    recipe: ["each token id (u32 LE)"],
  },
  {
    key: "media",
    name: "Media",
    color: "red",
    fields: "kind: MediaKind, bytes: Bytes",
    use: "Multimodal bytes resolved via Turn.content → endpoint",
    prefix: `"media\\0"`,
    recipe: ["kind string", "media bytes"],
  },
  {
    key: "tracehash",
    name: "TraceHashIds",
    color: "orange",
    fields: "hash_ids: Box<[i64]>, block_size: usize",
    use: "DynoSim / simulator adapters — cache identity only",
    prefix: `"trace-hash-ids\\0"`,
    recipe: ["block_size", "each hash id (i64 sequence)"],
  },
];

type DomainOptionProps = {
  domain: Domain;
  active: boolean;
  onSelect: () => void;
};

/**
 * Clickable domain-selector chip. `Chip`/`ChipNode` in `src/nodes/` is a `@xyflow/react`
 * diagram-node component and isn't usable outside a flow canvas, so this page renders a
 * plain button styled with the same surface/stroke/ink tokens Chip uses, plus a category
 * swatch and an active-state category tint.
 */
function DomainOption({ domain, active, onSelect }: DomainOptionProps): React.JSX.Element {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onSelect}
      className={clsx(
        "flex items-center gap-2 rounded-none border px-3 py-1 text-xs font-medium transition-colors",
        surfaceClassName(active ? "elevated" : "panel"),
        strokeClassName(active ? "primary" : "secondary"),
        active ? inkClassName("primary") : inkClassName("secondary"),
        active && categoryBgTintClassName(domain.color),
      )}
    >
      <Swatch color={domain.color} />
      {domain.name}
    </button>
  );
}

/**
 * Payload — six disjoint hash domains. Selecting a domain updates the displayed recipe
 * card, which lists the fields hashed together (after the shared version constant and
 * parent-id framing) to derive that domain's blake3 `SegmentId`.
 */
export function PayloadsPage(): React.JSX.Element {
  const [selectedKey, setSelectedKey] = useState<string>("message");
  const domain = DOMAINS.find((d) => d.key === selectedKey) ?? DOMAINS[0]!;

  const rows = [
    { label: "HASH_VERSION", sub: HASH_VERSION, faint: true },
    { label: "domain prefix", sub: domain.prefix, faint: false },
    { label: "parent id", sub: PARENT_ID_ROW, faint: false },
    ...domain.recipe.map((r) => ({ label: "├", sub: r, faint: false })),
  ];

  return (
    <Stack gap={16}>
      <div>
        <h2 className={clsx("text-lg font-semibold", inkClassName("primary"))}>
          Payload — six disjoint hash domains
        </h2>
        <p className={clsx("text-sm", inkClassName("secondary"))}>
          A segment&apos;s <code>Payload</code> is one of six variants. Each hashes under its own
          blake3 domain prefix, so the same bytes in two domains never collide. Every recipe is
          also framed by the parent&apos;s id and the version constant{" "}
          <code>b&quot;aiperf-dataset-segment-v1\0&quot;</code>. Select a domain to see its recipe.
        </p>
      </div>

      <Row gap={8} wrap>
        {DOMAINS.map((d) => (
          <DomainOption
            key={d.key}
            domain={d}
            active={d.key === selectedKey}
            onSelect={() => setSelectedKey(d.key)}
          />
        ))}
      </Row>

      <Grid columns="1fr 1fr" gap={14}>
        <div className={clsx("rounded-none border p-4", surfaceClassName("elevated"), strokeClassName("primary"))}>
          <div className="mb-3 flex items-center justify-between">
            <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>Variant</span>
            <span
              className={clsx(
                "rounded-none border px-2 py-0.5 text-[11px] font-mono",
                strokeClassName("secondary"),
                categoryClassName(domain.color),
              )}
            >
              Payload::{domain.name}
            </span>
          </div>
          <Stack gap={12}>
            <div>
              <div className={clsx("text-xs", inkClassName("tertiary"))}>Fields</div>
              <code className={clsx("mt-1 block text-xs", inkClassName("primary"))}>{domain.fields}</code>
            </div>
            <div>
              <div className={clsx("text-xs", inkClassName("tertiary"))}>Role in the pipeline</div>
              <p className={clsx("mt-1 text-sm", inkClassName("secondary"))}>{domain.use}</p>
            </div>
            <div>
              <div className={clsx("text-xs", inkClassName("tertiary"))}>SegmentDomain discriminant</div>
              <div className="mt-1">
                <code className={clsx("text-xs", inkClassName("primary"))}>
                  SegmentDomain::{domain.name}
                </code>{" "}
                <span className={clsx("text-xs", inkClassName("quaternary"))}>
                  drives dispatch, not field precedence
                </span>
              </div>
            </div>
          </Stack>
        </div>

        <div className={clsx("rounded-none border p-4", surfaceClassName("elevated"), strokeClassName("primary"))}>
          <div className="mb-3 flex items-center justify-between">
            <span className={clsx("text-sm font-semibold", inkClassName("primary"))}>blake3 recipe</span>
            <span className={clsx("text-[11px] font-mono", inkClassName("tertiary"))}>
              payload_id() · segment.rs:528
            </span>
          </div>
          <Stack gap={4}>
            {rows.map((r, i) => (
              <div
                key={`${r.label}-${i}`}
                className={clsx(
                  "flex items-center justify-between rounded-none border px-3 py-1.5",
                  r.faint ? strokeClassName("tertiary") : strokeClassName("primary"),
                  r.faint ? "opacity-70" : undefined,
                  r.faint ? surfaceClassName("panel") : surfaceClassName("elevated"),
                )}
              >
                <span className={clsx("text-xs font-bold", inkClassName("tertiary"))}>{r.label}</span>
                <span className={clsx("text-xs font-mono", inkClassName("primary"))}>{r.sub}</span>
              </div>
            ))}
            <div className={clsx("mt-2 text-center text-xs font-bold", categoryClassName(domain.color))}>
              ↓ blake3.finalize()
            </div>
            <div className={clsx("text-center text-xs font-mono", inkClassName("secondary"))}>
              SegmentId([u8; 32])
            </div>
          </Stack>
        </div>
      </Grid>

      <Callout tone="info" title="Why parent-by-id, not parent-by-index">
        A child hash includes its parent&apos;s content hash rather than its insertion index.
        Loading unrelated rows in a different order can&apos;t shuffle handles, so ids stay
        deterministic — and identical text under different prefixes stays distinct.
      </Callout>
    </Stack>
  );
}
