# Origin #23 finding: per-round authored DAG branches

Upstream `fc7bbf3bdd` adds request-free orchestrator conversations, repeated
and per-round authored branch fan-out, join barriers, per-round think-time,
payload-isolation mode, and associated loader/credit/phase accounting. The
change includes 23 unit-test files, five integration/component-integration
test files, and four DAG fixtures; it adds no Rust or upstream E2E test.

The native product has Graph-IR branch and join execution, chained-diamond
barrier ordering with think-time, AgentX join-gating parity, request-free
control stages, and native graph workload accounting. These seams deliberately
replace the Python DAG orchestrator rather than mirror its loader classes.
Focused native regressions passed: AgentX join-gating parity and both chained
diamond barrier tests (including the real mock endpoint). The ignored native
full-topology tests document the remaining metadata/export seam and were not
claimed as passing coverage.

No native implementation or test port is required for this superseded Python
DAG architecture. The behavior-level native coverage is already present.

Disposition: already-covered.
