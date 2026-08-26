# Commit 054 — `bfe33151de`

## Upstream intent

When filter-then-cap rejects every candidate for `--max-context-length`, report
the smallest observed peak context and the concrete limit that admits it. The
diagnostic must describe the completed selection pass, not invent a candidate
or alter selection order.

## Native analysis

The target is not Baseten. Native has three equivalent recorded-trace selection
seams:

- `agentx::selection` drives the legacy/HuggingFace WEKA filter-then-cap path;
- Graph-IR WEKA scans records sequentially when a root or context cap is set;
- Graph-IR Dynamo selects complete session trees by their largest request peak.

All three could reject every candidate while returning only a generic empty
selection error. The shared AgentX stats already carry scanned, rejected,
largest, eligible, and loaded counts, so the smallest observed peak belongs
there. Graph-IR owns independent selection loops because it must retain its
no-decode-after-cap and whole-tree invariants; it consumes the same diagnostic
contract without sharing parsing ownership.

TraceLab has no equivalent maximum-context filter in the native compiler. The
upstream Python TraceLab call site therefore has no native behavioral target.

## Required native behavior

1. Track the minimum peak in every scanned `agentx::selection` pass, including
   zero-valued peaks.
2. If the AgentX/HF pass is empty after a configured context cap, return an
   error containing source, scanned count, authored cap, optional root cap, and
   the exact smallest peak / actionable replacement cap.
3. If Graph-IR WEKA or Dynamo rejects every root because of a configured
   context cap, return the same actionable smallest-peak diagnosis while
   preserving each compiler's existing source terminology and selection order.
4. Empty sources and an empty selection for reasons other than an active
   context cap retain their existing errors; a root cap never appears as the
   alleged cause.
5. Cover helper behavior, AgentX/HF selection, WEKA compilation, Dynamo tree
   selection, and a native CLI/binary route which proves the error reaches the
   product boundary.

## Upstream-to-native test map

| Upstream behavior | Native evidence |
| --- | --- |
| minimum peak is recorded | `agentx::selection` unit test |
| all H/F WEKA rows rejected gives admission hint | `agentx::loader` test |
| all Graph-IR WEKA roots rejected | recorded-WEKA compiler test |
| all Graph-IR Dynamo trees rejected | recorded-Dynamo selector/compiler test |
| diagnostic reaches the public native command | Config-v2/native binary test with an inline or temporary WEKA input |
| no cap / empty source remains distinct | focused regression tests |

## Ancestry constraint

The final target-only merge records exact upstream
`bfe33151de75426710e51ca054823aa91342cebc` as its second parent. Its first
parent is the reviewed native Rust tree; upstream Python files are not imported
and no cherry-pick is used.
