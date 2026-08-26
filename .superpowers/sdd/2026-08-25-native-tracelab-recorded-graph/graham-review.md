# Graham Review — Native TraceLab Recorded Graph

Reviewed range: `f423b618da..604ddac599`

## Finding

### [P1] Preserve TraceLab source order after WEKA lowering

TraceLab requires stable source-session order because sequential root selection,
context filtering, and entry caps are order-sensitive. The native converter
created WEKA documents in source order, but the shared WEKA compiler restores
its historical deterministic output order by trace identifier after parallel
lowering. A corpus whose first sessions were `claude:z` and `claude:a`
therefore returned `[claude_a, claude_z]` rather than
`[claude_z, claude_a]`.

The regression was reproduced with a real JSONL compiler test before the repair.
Commit `cb264842ab` restores the converted order after WEKA lowering and avoids
cloning each complete source row into the timed-round view.

## Hot-path and systems audit

TraceLab conversion is a setup-time graph compilation path. The implementation
adds no request-path synchronization, tasks, channels, direct wall-clock reads,
or logging. Production error paths return contextual `RecordedTraceError`
values and contain no `unwrap()` or `expect()`. The graph adapter keeps the
existing acquired-source, segment-pool, and worker-local execution boundaries.
