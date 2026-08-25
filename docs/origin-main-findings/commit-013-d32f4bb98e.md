# Origin/main commit 013 finding: canonical speculative-decode acceptance metrics

Upstream `d32f4bb98edbeac1374ec816aee32d7e4517c5ae` adds canonical,
per-request speculative-decoding acceptance metrics. It consumes vLLM's
per-choice `speculative_decoding_stats`, retains one engine-neutral acceptance
record per request, derives six record distributions and five run-level
scalars, pools the accepted-draft histogram by benchmark phase, adds a dedicated
console section, and projects the canonical record and pooled histogram into
JSON artifacts.

The historical scout classified this commit as already covered by
`usage_accepted_prediction_tokens` and `usage_rejected_prediction_tokens`.
That comparison is not behaviorally valid. OpenAI prediction-token usage counts
describe accepted and rejected predicted output; they do not expose speculative
verification-step count, proposed draft volume, acceptance length, the
accepted-per-verified ratio, per-step acceptance distribution, or its pooled
histogram. This individual finding supersedes the provisional entry in
`early-002-030.md`.

The native product has no equivalent request-acceptance path. The existing
SGLang speculative-decoding console table reads server-wide Prometheus gauges
from `NativeReport.server_metrics`; it is a separate telemetry feature and
cannot substitute for request-attributed vLLM acceptance records. The upstream
commit is therefore **applicable, architectural, and not covered**. Its native
port spans OpenAI response decoding, the transport-neutral observer seam,
`RecordIngest`, exact/sketch/cellular metric folding, the native report,
per-record JSONL, the GenAI-Perf v1 JSON artifact, and console rendering.

The exact upstream diff adds unit and property tests but no integration or E2E
test file. There is therefore no upstream integration test to translate
literally. The required stronger native integration mapping is one deterministic
Rust mock-server fixture carrying the canonical worked example
`[2, 3, 1, 4, 2, 0, 3, 3]`, exercised by a real streaming `aiperf profile`
subprocess. The E2E asserts the dedicated console block, all scalar values and
the full pooled histogram in `*_aiperf.json`, and the canonical acceptance
record plus record metrics in `profile_export.jsonl`. A second real profile
against the ordinary mock proves that absent stats suppress the console block,
summary metrics, histogram, and per-record field.

## Integration ancestry

The required non-fast-forward merge is
`e93d959c62af971cf867ef54c98c608452ade195`. Its parents are the controller's
exact shared #12 head `8b5194bcfc26475c5e06030d8701c82b66eb7b6a`
and exact upstream commit `d32f4bb98edbeac1374ec816aee32d7e4517c5ae`.
The four synchronized agent-instruction conflicts retained the native branch's
current architecture body; the upstream metric implementation and tests were
otherwise preserved. The merge commit used `--no-verify` because this isolated
worktree has no `.venv`; equivalent validation must run through the shared
project environment before closure.

The native design is recorded in
[`../specs/2026-08-25-native-spec-decode-acceptance-metrics.md`](../specs/2026-08-25-native-spec-decode-acceptance-metrics.md).
