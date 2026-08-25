# Origin #17 closure: idle cap after barrier deferral

Upstream commit `c02d02db28` changes the Python AgentX scheduler so the global
idle-gap cap is re-evaluated when a scheduled task is retained by a
cross-stream replay barrier and the scheduler later drains. It adds a
`LoopScheduler` drain observer, elapsed-idle accounting, and a focused
barrier-retention regression. The upstream change has no integration or
end-to-end tests beyond its unit coverage.

Native recorded replay uses a different, synchronous scheduler architecture.
The equivalent guard is evaluated when a continuation is scheduled, using the
runtime scheduler task count: `system_idle_continuation_delay_ms` caps the
delay only when no other scheduled work remains. Native replay's tree gate and
replay barrier separately retain and release cross-stream turns, and their
unit tests cover deferred joins and release after child completion.

Native coverage is already present in
`rust/runtime/src/agentic_replay.rs`:

- `system_idle_continuation_caps_only_when_no_other_tasks_are_pending` covers
  the pending-task versus drained-task cap decision.
- `tree_gate_defers_join_until_children_terminal` and
  `join_gating_defers_and_releases_parent` cover the barrier-retention and
  release path that feeds that decision.

Because the native scheduler does not have Python's asynchronous drain
observer seam, no direct implementation port is required. The behavior-level
coverage is retained in the native architecture, and no native integration/E2E
test is applicable.

Disposition: already-covered; exact merge performed for campaign ancestry.
