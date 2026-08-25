# Origin #14 closure: cache-bust enables dataset wrapping

Upstream commit `4fe3ff7154` changes the Python trajectory wrap policy so an
enabled cache-bust target is an implicit opt-in for wrapping an oversubscribed
recorded corpus. It adds two policy cases and two timing-configuration ordering
cases; it does not add an integration or end-to-end test.

This behavior is already implemented in native Graph-IR execution. The graph
phase runner passes `allow_dataset_wrap || cache_bust_enabled` to its wrap
validator, and the offline and stdio entrypoints propagate the cache-bust
state. The validator retains the upstream one-pass and in-corpus session
exceptions and reports cache-bust as an alternative remediation.

Native coverage is in
`rust/runtime/src/engine/graph_phase_runtime.rs`:

- `recorded_graph_wrap_policy_rejects_unintentional_lane_cloning` verifies the
  disabled policy rejects oversubscription.
- `recorded_graph_wrap_policy_allows_wrapping_when_cache_bust_is_enabled`
  verifies an enabled target permits oversubscription.
- `recorded_graph_wrap_policy_allows_bounded_or_one_pass_corpora` retains the
  existing exceptions.

The separate native config validation tests cover scenario cache-bust
resolution, including the disabled default and scenario-selected target. No
additional Rust implementation or upstream integration-test port is required.

Disposition: already-covered; exact merge performed for campaign ancestry.
