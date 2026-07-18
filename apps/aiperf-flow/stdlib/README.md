# AIPerf Flow stdlib (P0 stubs)

This tree holds the first P0 standard-library and flagship domain symbol stubs for
`apps/aiperf-flow`.

These files are **authoring placeholders** pending expansion into full stdlib
compositions, golden IR fixtures, and compile parity. Each stub is a valid
`language 1` document with a `symbol` definition and a minimal scene shell.

## P0 flagship stubs

| File | Symbol | Wraps (target) |
| --- | --- | --- |
| `TokenSpanMorph.flow` | `TokenSpanMorph` | `SpanMap`, `SemanticMorph` |
| `PromptSegmentComposer.flow` | `PromptSegmentComposer` | `SegmentStrip` |
| `RequestLifecycleWaterfall.flow` | `RequestLifecycleWaterfall` | `Waterfall` |

Symbol bodies currently use simple `ComponentInvocation` placeholders only. Typed
array params, slots, timeline anchors, and stdlib barrel exports will land in
follow-up stdlib work.
