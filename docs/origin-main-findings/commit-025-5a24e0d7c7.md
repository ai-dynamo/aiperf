# Origin commit 5a24e0d7c7: average error-rate SLA

## Finding

The upstream change fixes `max-concurrency-under-sla` error-rate evaluation by
using the run-level average and converting the public fractional flag to the
native metric's percentage-point unit. Its coverage is one slow integration
test (`tests/integration/test_search_recipes.py`) and two unit-test files
(`tests/unit/metrics/test_request_error_rate_metric.py` and
`tests/unit/search_recipes/test_max_concurrency_under_sla.py`); it adds no E2E
test. The Rust metric catalog already emits `request_error_rate` as percentage
points, but the native search builder incorrectly requested `p99` and compared
the unscaled fraction.

## Specification and paired scope

For `--error-rate-sla f`, native `max-concurrency-under-sla` must emit one
`request_error_rate:avg:lt:(100*f)` filter. Zero-error runs must therefore be
feasible and retain their search boundary. The related upstream #48 commit
(`260d00f5e9`, adaptive error-rate units) changes adaptive-scale documentation
and parsing/tests; it is reviewed as a coupled follow-up but is intentionally
not merged or implemented by this port.

## Test audit and closure

The upstream unit assertions map to a Rust unit test for filter statistic and
unit conversion. The upstream integration scenario maps to the existing native
search/SLA path; no separate Rust E2E test is applicable because native search
is exercised through CLI unit/post-processing seams in this branch and the
upstream test launches the Python CLI. Focused Rust tests and formatting pass.

Graham review found no issues: the change is confined to the search filter
seam, performs no async or hot-path allocation, and adds no unwrap/expect or
synchronization. The initial broad test command exhausted `/tmp` while
compiling unrelated integration binaries; the narrowed lib-only test then
exposed and avoided a stack-overflowing `ProfileFlags` fixture. The untouched
crash core remains untracked in the worktree.
