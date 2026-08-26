# Native rejected peak-context diagnostics

## Goal

Make every native WEKA (including its TraceLab adapter) and Dynamo recorded-trace selection report the smallest
observed peak context when a configured maximum rejects every eligible root.

## Contract

Selection continues to filter before applying its root cap. It scans only the
documents it would otherwise scan. While scanning, it retains a minimum peak
that is initialized by the first candidate rather than by a sentinel so a
valid zero peak remains representable.

When a non-empty scan with `max_context_length = L` retains no candidate, the
error names the source and states `Smallest trace requires M tokens; raise
--max-context-length to at least that (e.g. --max-context-length M) to admit
any trace.` Here `M` is the minimum peak over exactly the scanned candidates.
For Dynamo, a candidate is a whole root session tree and its peak is the
maximum request peak inside that tree. For WEKA, it is the trace peak already
used by the existing filter, honoring top-level `max_osl` and uncapped
subagents.

The hint is emitted only when the context filter caused the empty result.
Empty sources, zero roots for unrelated malformed/input reasons, and ordinary
root limiting retain their established failures. No selector decodes after its
existing cap boundary merely to improve a diagnostic.

## Design

`agentx::selection::SelectionStats` gains `smallest_observed: Option<i64>` so
zero is unambiguous. A formatter builds the one stable contextual tail from
the stats and is used by `load_hf_traces_from_rows` after filter-then-cap.
The helper accepts a format-specific source label and uses the authored caps
only as metadata; it does not decide whether a selection was causally empty.

Graph-IR WEKA and Dynamo retain their local parsers/selectors. The TraceLab
adapter supplies a TraceLab source label when it delegates to WEKA. The loops add
a local smallest-peak accumulator only in the context-filter branch and use a
small shared `RecordedTraceError` constructor after exhaustion. This keeps
parsing, ordering, tree grouping, and no-decode-after-cap boundaries local.

## Verification

Unit tests exercise a zero peak and ordinary minimum selection. Legacy/HF
WEKA, TraceLab, Graph-IR WEKA, and Dynamo tests use unequal rejected peaks and assert the
minimum plus the exact suggested flag value. A real native command test drives
the Graph-IR WEKA source through the public Config-v2 execution surface and
asserts the process error contains the same actionable tail. Tests also retain
the existing empty-source error and prove a successful filtered selection is
unchanged.

## Closure verification

`4022b433c9` passed the AgentX focused suite (106 passed, 1 ignored), the
recorded-graph suite (78 passed, 1 ignored, with the separate one-test source
acquisition harness also passing), and the native CLI E2E (1 passed). The E2E
asserts that an unreachable endpoint has no connection attempt because the
terminal diagnostic is produced during selection. Formatting and both changed-
scope Clippy targets pass. The broad runtime library suite has one unrelated
version-snapshot failure in `metrics_core::report`; it is not in this port's
diff. Independent Graham review approved the implementation without findings.
