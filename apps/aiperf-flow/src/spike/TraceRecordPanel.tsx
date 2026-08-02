/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the `dynamo.request.trace.v1` record for the turn being built, as live JSON.
//!
//! Field names and nesting follow `rust/runtime/src/graph/recorded/dynamo/schema.rs`:
//! `schema`, `event_type`, `event_time_unix_ms`, `agent_context`, and `request` with a nested
//! `replay` carrying `trace_block_size`, `input_length` and `input_sequence_hashes`.
//!
//! `input_sequence_hashes` is rendered as blocks rather than text, because the point is *which*
//! of them were already known. A block lights green the moment the growing turn reaches it and it
//! turns out to be a repeat — a KV prefix hit — and orange when it is new.

import { BLOCK_SIZE, type TraceTurn } from "./segmentSim.js";

const KEY = "var(--color-category-cyan)";
const STR = "var(--color-category-green)";
const NUM = "var(--color-category-yellow)";

function K({ k }: { k: string }) {
  return <span style={{ color: KEY }}>&quot;{k}&quot;</span>;
}
function S({ v }: { v: string }) {
  return <span style={{ color: STR }}>&quot;{v}&quot;</span>;
}
function N({ v }: { v: number }) {
  return <span style={{ color: NUM }}>{Number.isInteger(v) ? v : v.toFixed(1)}</span>;
}

export function TraceRecordPanel({
  turn,
  parentSessionId,
}: {
  turn: TraceTurn | null;
  parentSessionId: string | null;
}): React.JSX.Element {
  const hit = turn?.hashes.filter((h) => h.reused).length ?? 0;
  const total = turn?.hashes.length ?? 0;

  return (
    <div className="rounded-lg border border-white/10 bg-surface-panel px-4 py-2.5">
      <div className="mb-1.5 flex items-baseline gap-3">
        <span className="text-[12px] font-bold tracking-widest text-ink-tertiary">
          DYNAMO TRACE RECORD — built as the turn accumulates
        </span>
        {total > 0 && (
          <span className="text-[13px] tabular-nums text-ink-quaternary">
            {hit}/{total} blocks already cached · {turn?.cachedTokens ?? 0} cached tokens
          </span>
        )}
      </div>

      {turn === null ? (
        <code className="block font-mono text-[13px] text-ink-quaternary">
          Waiting for the first request_end record…
        </code>
      ) : (
        <code className="block overflow-x-auto font-mono text-[13px] leading-[1.5] text-ink-secondary">
          <div>{"{"} <K k="schema" />: <S v="dynamo.request.trace.v1" />,</div>
          <div className="pl-3">
            <K k="event_type" />: <S v="request_end" />, <K k="event_source" />: <S v="dynamo" />,
          </div>
          <div className="pl-3">
            <K k="event_time_unix_ms" />: <N v={turn.eventTimeMs} />,
          </div>
          <div className="pl-3">
            <K k="agent_context" />: {"{"} <K k="session_id" />: <S v={turn.sessionId} />
            {parentSessionId !== null && (
              <>
                , <K k="parent_session_id" />: <S v={parentSessionId} />
              </>
            )}{" "}
            {"}"},
          </div>
          <div className="pl-3">
            <K k="request" />: {"{"} <K k="request_id" />: <S v={turn.requestId} />,{" "}
            <K k="input_tokens" />: <N v={turn.inputLength} />, <K k="output_tokens" />:{" "}
            <N v={turn.outputTokens} />,
          </div>
          <div className="pl-6">
            <K k="cached_tokens" />: <N v={turn.cachedTokens} />, <K k="ttft_ms" />:{" "}
            <N v={turn.ttftMs} />, <K k="total_time_ms" />: <N v={turn.totalTimeMs} />,
          </div>
          <div className="pl-6">
            <K k="replay" />: {"{"} <K k="trace_block_size" />: <N v={BLOCK_SIZE} />,{" "}
            <K k="input_length" />: <N v={turn.inputLength} />,
          </div>
          <div className="flex flex-wrap items-center gap-1 py-1 pl-9">
            <K k="input_sequence_hashes" />
            <span>: [</span>
            {turn.hashes.length === 0 && (
              <span className="text-ink-quaternary">
                &nbsp;— under one block so far&nbsp;
              </span>
            )}
            {turn.hashes.map((h) => (
              <span
                key={h.index}
                title={`block ${h.index} · tokens ${h.index * BLOCK_SIZE}–${(h.index + 1) * BLOCK_SIZE}\n${h.reused ? "already cached (prefix hit)" : "new block"}`}
                className="rounded px-1.5 py-[3px] text-[13px] font-semibold tabular-nums"
                style={{
                  // Lit as the turn reaches it: green means this exact block was already
                  // known, so its tokens land in `cached_tokens` and never get recomputed.
                  background: h.reused ? "var(--color-category-green)" : "var(--color-category-orange)",
                  color: "#000",
                }}
              >
                {h.value}
              </span>
            ))}
            <span>]</span>
          </div>
          <div className="pl-6">{"} } }"}</div>
        </code>
      )}

      <div className="mt-1 flex items-center gap-4 text-[13px] text-ink-quaternary">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: "var(--color-category-green)" }} />
          reused — prefix already cached
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: "var(--color-category-orange)" }} />
          new — this block must be computed
        </span>
        <span>one block = {BLOCK_SIZE} tokens</span>
      </div>
    </div>
  );
}
