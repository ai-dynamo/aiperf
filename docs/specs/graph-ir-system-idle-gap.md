# Graph-IR System Idle-Gap Cap

## Goal

Apply `system_idle_gap_cap_seconds` to Graph-IR dispatch timing so the graph does not remain globally idle for an oversized recorded delay. Existing `idle_gap_cap_seconds` trace warping remains unchanged.

## Design

Graph-IR computes node firing gates in `graph::executor::TraceExecutor::compute_firing_gate_us`, after input readiness and causal edge constraints are known. The executor receives the system idle cap in milliseconds and applies it at this boundary. The cap compresses waits only when the graph has no other active work; branch timing and causal ordering remain intact.

The configuration is threaded from the protocol-v2 graph workload configuration through graph executor construction. `None` preserves current behavior. Non-positive or non-finite values are treated as disabled by existing config validation/normalization.

## Testing

Add deterministic simulation coverage for an oversized global idle interval, a below-cap interval, an unset cap, and concurrent branches. Add protocol/config propagation coverage and run the focused Graph-IR tests plus a 062126 dry-run.
