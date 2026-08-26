# origin/main Rust Porting Campaign Ledger

## Scope and authority

This is the authoritative closure record for the campaign started from
`ajc/rust-merge-main` at `0b67e08284ef`, compared with `origin/main` pinned to
`c2889280a66fc85b44e9456fd7020874c73a44fc`. The historical inventory contains
60 commits (`git rev-list --reverse --topo-order 0b67e08284ef..c2889280a66f`);
`817a8d84ddb9` is already merged, leaving 59 main-only commits.

`docs/origin-main-findings/` contains supporting source scouts, not closure
records. A disposition records a source comparison; it never proves a merge,
implementation, test run, or review.

## Campaign fields

- **Campaign state:** `pending`, `analysing`, `merged`, `rust-porting`, or
  `complete`. Only `complete` means merge ancestry, applicable implementation,
  current verification, and Graham review are recorded here.
- **Disposition:** `unassessed`, `already-covered`, `applicable`,
  `not-applicable`, `superseded`, or `shared-product-action`. The last category
  covers shipped-image, packaging, or release work that reaches the product but
  is not a Rust-source port.

## Campaign inventory

| # | Upstream commit | Subject | State | Disposition | Evidence / next action |
| ---: | --- | --- | --- | --- | --- |
| 1 | `817a8d84ddb9` | LCB out-of-process grading | complete | applicable | Merged in `1c03271dac3e`; native delegation/reaping/detail commits and verification below. Later concurrency parity is tracked by #35. |
| 2 | `0883bd1aee` | version 0.12.0 | merged | not-applicable | Python metadata/docs only. |
| 3 | `34b2be2ee1` | SGLang speculative metrics | complete | applicable | Merged in `4d076c660f`; native console policy/renderer/mock/E2E commits and verification below. Graham approved. |
| 4 | `6db948524e` | SageMaker malformed timestamp | merged | already-covered | Native loader returns validation error. |
| 5 | `ce715ae849` | agentic think-time idle guard | complete | applicable | Merged in `6e8da730e8`; native scenario default/lock port in `ad4c2a54f4`, `6e78b41d35`, and `729c59f64a`; Graham approved. |
| 6 | `93b6223373` | telemetry field rename | complete | applicable | Merged in `4ab850c79d`; native artifact migration and custom-prefix closure in `726fcb614b`, `4763b53f0f`, and `978bc93fea`; Graham approved. |
| 7 | `86ea3f7deb` | detailed JSONL fallback | complete | already-covered | Merged in `f027104364`; native detailed aggregation already reads the canonical JSONL path directly, while the compatibility Python regression passes. Graham approved. |
| 8 | `5566aae1e1` | ShareGPT batch encoding | complete | applicable | Actual upstream merge `6521729344`; native ordered batch seam and bounded ShareGPT reconstruction in `4e39be3aee` / `67e6f3988d`; independently Graham-approved on current shared HEAD. Closure: `docs/origin-main-findings/commit-008-5566aae1e1.md`. |
| 9 | `844efe1b36` | full synthesis prefix blocks | complete | applicable | Merged in `cb1d016398`; native implementation preserves whole prefix blocks and both synthesize-to-profile mock-server E2Es are green. |
| 10 | `ffc943a9fe` | FFmpeg CVE bump | complete | shared-product-action | Merged in `830f579589`; coupled FFmpeg 8.1.2 attribution contract test is green and reviews approved. Test-image build reached FFmpeg then exposed the unrelated missing `COPY contracts` wheel-builder defect; no rebuilt-image claim. |
| 11 | `d55ae21d34` | ShareGPT tutorial timeout | merged | not-applicable | Merged in `4d52f494e3`; documentation-only cold-cache timeout guidance, no native test candidate. Closure: `docs/origin-main-findings/commit-011-d55ae21d34.md`. |
| 12 | `8148999496` | LCB trust_remote_code | complete | already-covered | Merged in `abd532d4b9`; the Rust-launched evaluator uses its script-free pinned raw-file loader. Upstream has no integration/E2E test to port. Closure: `docs/origin-main-findings/commit-012-8148999496.md`. |
| 13 | `d32f4bb98e` | canonical spec-decode metrics | complete | applicable | Merged in `ee16a9aa82`; native vLLM metrics, artifacts, mock fixture, Rust E2E, and Graham review approved. |
| 14 | `4fe3ff7154` | cache-bust enables wrapping | complete | already-covered | Merged in `0cc26e0c82`; native graph wrap policy covers it. |
| 15 | `2eb04aa2f8` | MMVU docs memory | complete | not-applicable | Merged in `3bfb5271c2`; documentation-only. |
| 16 | `5ad08166a1` | profiling grace default | complete | applicable | Merged in `dbcb9b53cf`; native duration grace default port reviewed. |
| 17 | `c02d02db28` | idle cap after barrier | complete | already-covered | Merged in `c8b0935acc`; native continuation guard regression reviewed. |
| 18 | `2f413f0dec` | minified schemas | complete | shared-product-action | Merged in `aad2729477`; generated schema minification verified. |
| 19 | `9b5f5f7282` | warmup prefix stand-down | complete | superseded | Merged in `d014d9f8a3`; closure records #32 ownership. |
| 20 | `446d2cd4b3` | empty raw_messages delta | complete | applicable | Merged in `38cba410fd`; native endpoint behavior and integration coverage reviewed. |
| 21 | `20eb25626a` | docstring policy | complete | not-applicable | Merged in `c56a407c25`; contributor-doc guidance only. |
| 22 | `1d1829540b` | structured cache-bust | complete | already-covered | Merged in `7bb2f67882`; native structured coverage reviewed. |
| 23 | `fc7bbf3bdd` | DAG branches | complete | already-covered | Merged in `26c5f285d9`; native Graph-IR/AgentX branch and join regression reviewed. |
| 24 | `f8c8e36533` | BranchStats test | complete | not-applicable | Merged in `89b5862670`; Python-only test correction, native phase regression reviewed. |
| 25 | `5a24e0d7c7` | average error-rate SLA | complete | applicable | Merged in `ca54427a94`; native adaptive SLA projection and search tests reviewed. |
| 26 | `093038afab` | JSON-array endpoint responses | complete | applicable | Merged in `6cd8137681`; native KServe JSON-array response parsing reviewed. |
| 27 | `f0128e105b` | reset/profiler hooks | complete | already-covered | Merged in `fd3fd13087`; native lifecycle policy and E2E coverage reviewed. |
| 28 | `aaaa72e69d` | arrival tutorial flag | complete | not-applicable | Merged in `5c7fe69bf5`; documentation-only. |
| 29 | `c2f5e9d459` | Python enum cache | complete | not-applicable | Merged in `d13e303b2f`; Python-specific hot path. |
| 30 | `9be59a9636` | AMD telemetry tests | complete | already-covered | Merged in `2cd8de8d01`; native telemetry fixture lifecycle and CLI collector validation cover the applicable behavior. |
| 31 | `59cd43e43a` | Responses mock model | complete | already-covered | Merged in `acf367397b`; native typed route/model/token/recording coverage already exists and a raw-record Rust E2E is green. |
| 32 | `1b34de0637` | warmup-isolation cache-bust | complete | applicable | Merged in `137b2f641d`; native phase-gated cache-bust target selection is covered by 12 runtime tests. |
| 33 | `37467dc38c` | force content parts | complete | applicable | Merged in `73e1ab0761`; native bootstrap-captured endpoint policy and focused test coverage are green. |
| 34 | `00ba1c5db3` | audio transcription | complete | applicable | Native multipart endpoint, strict parser/format validation, mock route, and real-binary E2E; final Graham review recorded in the finding. |
| 35 | `03c9c6ddc5` | LCB grading concurrency | complete | applicable | Merged exactly in `7cd1a5bf29`; native response-id demultiplexing, persistent reader-fault state, shutdown/drop process-group cleanup, and a real Rust-launched batch/reap integration are green and Graham-approved after fixes in `429050fbf0`. |
| 36 | `6480e5467f` | synthesis YAML overrides | complete | applicable | Merged in `445e2ec4d7`; native CLI-over-YAML overlay, resolver and CLI-path integration tests, exact upstream merge ancestry, and Graham approval recorded. |
| 37 | `23ed221c3d` | CI speed | complete | not-applicable | Exact upstream merge completed on `ajc/port-origin-037`; Python harness regression passed and no Rust test candidate exists. Closure: `docs/origin-main-findings/commit-037-23ed221c3d.md`. |
| 38 | `c26fe88bd8` | AgentX FAQ | complete | not-applicable | Merged in `233b71c98b`; closure: `docs/origin-main-findings/commit-038-c26fe88bd8.md`. |
| 39 | `1e32a51318` | Baseten load performance | complete | applicable | Exact merge `da917561fb`; projected typed native loader, 17 focused tests, public adapter parity, measured 19.4% median time / 25.1% median RSS reduction, and independent Graham approval recorded. |
| 40 | `215be05b6a` | Baseten outcome fidelity | complete | applicable | Merged exactly in `1d20f63c51`; native E2E, TTFT, and cached-token outcomes survive real-Parquet registry composition without dispatch leakage and are Graham-approved. |
| 41 | `516faa12c8` | CODEOWNERS | complete | not-applicable | Exact merge in `eeb59a96d5`; closure note `docs/origin-main-findings/commit-041-516faa12c8.md`. |
| 42 | `ce453582c7` | CONTRIBUTING spelling | complete | not-applicable | Exact merge completed on `ajc/port-origin-042`; closure note `docs/origin-main-findings/commit-042-ce453582c7.md`. |
| 43 | `6ed4823d12` | cache-bust help link | complete | not-applicable | Exact merge completed in `e9767333b6`; closure note `docs/origin-main-findings/commit-043-6ed4823d12.md`. |
| 44 | `082a51827e` | TraceLab dataset | complete | applicable | Native plain/gzip TraceLab-to-Graph-IR conversion, config/CLI/cellular integration, real-binary coverage, exact target-only merge ancestry, and Graham approval recorded. |
| 45 | `21f8ad7b3e` | high-resolution pacing | complete | applicable | Exact merge `86a93aaec1`; native bounded local/sharded catch-up, exact 5,000-request real-clock evidence, and Graham approval recorded. |
| 46 | `e659d2a95a` | FFmpeg codec allowlist | pending | shared-product-action | Audit native codec defaults before porting shared Dockerfile policy. |
| 47 | `9e96b499d1` | aiohttp minimum | pending | already-covered | Bundled Python bound already satisfies it. |
| 48 | `260d00f5e9` | adaptive error-rate units | pending | applicable | Adaptive SLA differs from exported percentage/cancellation contract; pair with #25. |
| 49 | `88242293b5` | verbatim system prompts | complete | applicable | Exact merge `9eeeac98f9`; native one-time file/CLI resolution, BLAKE3 composition identity, OpenAI/Anthropic wire E2E, exact 40-test mapping, and Graham approval recorded. |
| 50 | `ade1f69eb1` | seamless phase transitions | complete | applicable | Exact merge `8252633121`; incoming lowering plus local/cellular first-owner-start/last-owner-stop profiler coordination, focused integration coverage, and two Graham approvals recorded. |
| 51 | `324bb05773` | per-chunk usage | complete | applicable | Exact target-only merge `7520bbefcc`; native authoring, response reduction, corrected primary/adaptive metrics, both OpenAI mock streaming routes, mandatory exact-upstream Python/native E2E, inherited-suite classifications, and independent two-pass Graham approval are recorded in `docs/origin-main-findings/commit-051-324bb05773.md`. |
| 52 | `810fd8bdd4` | vLLM spec-decode wire | complete | applicable | Exact target-only merge `9fcd17c62e`; native root metrics, dense histogram normalization, automatic trailing-usage negotiation, four-mode mock/product integration, and 15/15 E2E are complete. Independent Graham review approved `082457b2de`. |
| 53 | `e5ebe915df` | enum inequality | pending | not-applicable | Python enum behavior only. |
| 54 | `bfe33151de` | rejected peak diagnostics | complete | applicable | AgentX/HF, WEKA (including TraceLab), and Dynamo selection; target-only merge `352ca1b032`, implementation `4022b433c9`, independent Graham approved. |
| 55 | `dd3f09b0c3` | session-affinity header | pending | applicable | Add default-on `X-Session-Affinity`, not opt-in `X-Session-ID`. |
| 56 | `94fee7338b` | random range ratio | complete | applicable | Exact merge `cd31c0ae5a`; native PCG64/MT19937 stream, config, prefix/special-token semantics, and 48 byte-exact production captures are covered by three audits and independent Graham approval. |
| 57 | `e10d53b1d3` | version 0.13.0 | pending | shared-product-action | Coordinated release decision, not standalone Rust port. |
| 58 | `9b60a3d479` | FakeTokenizer | pending | not-applicable | Python test harness only. |
| 59 | `c9288da6c1` | mmap conversation cache | pending | not-applicable | No equivalent native mmap backend. |
| 60 | `c2889280a6` | random_pool batch sizes | complete | applicable | Native four-modality CLI/YAML/config projection, safe real-loader batching, and binary integration coverage approved at `5dd2939765`; exact target-only merge `f1d39ad583`. |

## Per-commit record: 817a8d84ddb9

### Upstream intent

Move LiveCodeBench code-generation grading to a supervised child process so a
wedged grader does not hold the parent benchmark process.

### Closure evidence

`1c03271dac3eb6465538dabf6950fd255baeac7d` is a two-parent merge with
`817a8d84ddb90d1e12c2a03327e16d853bb4e6e0` as second parent. Native delegation
and lifecycle work landed in `63eb01a355`, `e62454bfcd`, and `9e859f6110`.
The final commit preserves finite nested `detail.pass@1` scores across the
child JSONL boundary.

The SDD record is `.superpowers/sdd/2026-08-25-native-lcb-codegen-worker/`.
Its Graham re-review is approved with no Important or Critical findings.
Focused Python verification reported 53 passed, 2 skipped, and 2 deselected;
`cargo test -p aiperf-runtime --features engine --lib accuracy_core::worker`
reported 7 passed using a target directory under `/mnt/4tb`.

This closes the original out-of-process-worker port. It does not assert that
later upstream request multiplexing and leader-exit descendant reaping (#35)
are equivalent; those are separately pending.

## Per-commit record: 34b2be2ee115

### Upstream intent and Rust comparison

Upstream adds a dedicated SGLang speculative-decoding console table with
configured-model and scheduler-leader selection, per-series identity,
finite-summary enforcement, percent-only display scaling, and inactive-gauge
suppression. Native server-metric collection already retained the required
gauges, labels, endpoints, and summaries, while the native SPEED-Bench reader
recognized their names only for explicit report generation. Ordinary native
profile console output did not consume server metrics, so an applicable Rust
port was required.

### Merge and implementation evidence

`4d076c660f31d9a9bf66f839867c3b9737e1a0ba` is the existing two-parent merge
whose second parent is exact upstream commit
`34b2be2ee1159cc7e6985e027027791d18dad693`; no new upstream merge was added.
The finding and design record landed in `456801aa8a`. Internal configured-model
projection landed in `debc81e3ef`; the native console renderer landed in
`67fd3896f9`; deterministic native mock gauges and real-profile Rust E2E coverage
landed in `91aae2ee05`, with exact row/cell assertions in `ae7c592f99`; and
`475352c0d3` records the mock fixture in its operating reference.

The renderer preserves case-insensitive configured-model matching,
absent-or-zero PP/TP leader selection, endpoint and varying-label distinctions,
finite avg/min/max/p50/p90 summaries, one-decimal percent display for raw rate
ratios, two-decimal accepted lengths, and length-first inactivity suppression.
It borrows `NativeReport`, so the raw server-metrics exports remain unchanged.

### Verification and review

The exact upstream commit adds one Python unit-test module and no integration or
E2E test. The native strengthening launches the pinned real Rust `aiperf`
binary against the in-process Rust mock, verifies exact `75.0`/`2.50` console
row cells, and separately verifies raw JSON rate `0.75` plus the model and
leader labels.

Fresh verification with `/usr/bin/sccache` and a Cargo target under `/mnt/4tb`
reported 24 console tests, 7 config-export tests, 1 native profile E2E, and 3
mock Prometheus tests passed. Debug CLI/mock and release CLI builds passed;
runtime all-target engine Clippy exited successfully. Scoped formatting and the
complete range diff check passed; the only workspace-format result is an
unrelated pre-existing `sidecar_input.rs:787` wrapping difference.

Independent task and whole-range reviews are clean after their focused fixes.
The final systems review reports zero Critical, Important, or Minor findings
and ends `GRAHAM APPROVED`. The specification is
`docs/specs/2026-08-25-native-sglang-speculative-console.md`; detailed evidence
lives in
`.superpowers/sdd/2026-08-25-native-sglang-speculative-console/`.

## Per-commit record: ce715ae849e5

### Upstream intent and Rust comparison

Upstream replaces AgentX MVP's per-trace compression with a ten-second global
idle guard that preserves trace-local think time and forbids incompatible
legacy timing caps. Native schedulers already implement global guard behavior,
but `inferencex_agentx_mvp()` still injected a per-trace cap and the native
scenario lock did not reject authored trace or inter-turn caps. This required
a native scenario/resolver port.

### Merge and implementation evidence

`6e8da730e816683350bf4be09942af755c422791` is a two-parent merge with
`ce715ae849e54d8b37141e7770aeadfc60985302` as its second parent. Native
policy and pure-lock work landed in `ad4c2a54f4`; resolver projection and
coverage landed in `6e78b41d35`; `729c59f64a` supplies the Graham-requested
format-only correction. The design spec is
`docs/specs/2026-08-25-native-agentx-global-idle-guard.md`, and the Sol plan
and SDD evidence live under `.superpowers/sdd/2026-08-25-native-agentx-global-idle-guard/`.

### Verification and review

Fresh focused scenario policy verification reported 8 passed, and the native
CLI resolver selector reported 2 passed using `/usr/bin/sccache` with a Cargo
target below `/mnt/4tb`. Graham additionally exercised system-idle unit tests
(3 passed) and real-binary E2E coverage (14 passed). The final re-review is
`GRAHAM APPROVED` with no Important or Critical findings.

## Per-commit record: 93b622337315

### Upstream intent and Rust comparison

Upstream replaces the vendor-specific public telemetry source field
`dcgm_url` with `telemetry_source_url` throughout the telemetry dataflow and
JSONL contract, without a compatibility alias. Native Rust already retained
collector-neutral source identity internally as `GpuTelemetryRecord.endpoint_url`,
but its public `TelemetryRow` serializer and product assertions still emitted
the obsolete key.

### Merge and implementation evidence

`4ab850c79dceb3fae8995b7c2a83550d6450d5cc` is a two-parent merge with exact
upstream commit `93b622337315a945d0f42511fefb314c8a1ff085` as its second parent.
The native serializer, runtime/product assertions, and telemetry design record
landed in `726fcb614b8320e87f955649fffcb6baec783489`; the serializer maps the
unchanged `record.endpoint_url` to exactly `telemetry_source_url` and never
dual-writes `dcgm_url`. Graham's first review made the custom-prefix artifact
lookup mandatory in `4763b53f0f0766d550b79c25958938983a710794`. That stronger
test exposed the native resolver dropping the documented GPU telemetry prefix;
`978bc93fea50fa941f7e34cc1dc5812c4b0c646f` now projects the normalized stem
to `<stem>_gpu_telemetry.jsonl` while preserving the no-prefix default.

The design spec is
`docs/specs/2026-08-25-native-telemetry-source-url.md`; the Sol plan and SDD
evidence live under
`.superpowers/sdd/2026-08-25-native-telemetry-source-url/`.

### Verification and review

The merged Python telemetry model/writer tests reported 88 passed. Native
verification using `/usr/bin/sccache` and a Cargo target under `/mnt/4tb`
reported 5 focused runtime telemetry tests, 1 focused CLI resolver test, and
17 real-binary GPU telemetry E2E tests passed; the release CLI build and direct
`rustfmt` checks on all four changed Rust files also passed. The final
independent task review found zero Critical, Important, or Minor findings. The
fresh Graham re-review found no Critical or Important findings, confirmed both
earlier findings resolved, and ended `GRAHAM APPROVED`; its sole documentation
whitespace minor was removed before closure.

## Per-commit record: 86ea3f7deb74

### Upstream intent and Rust comparison

Upstream fixes the legacy Python `cli_runner` detailed aggregator so an absent
`export_jsonl_file` uses `DEFAULT_JSONL_FILENAME` instead of an empty path
component. The native `aiperf profile` path does not call that Python helper.
Its Rust-owned detailed sweep aggregation reads
`profile_export.jsonl` directly in `rust/cli/src/sweep/aggregate.rs`, so the
native behavior was already equivalent and no Rust source change or design
specification was needed.

### Merge, verification, and review

`f027104364e19361607a156a57b4051901abf62c` is a two-parent merge with exact
upstream commit `86ea3f7deb74ef49ae84a1bee293eb724125788c` as its second
parent. It retains the upstream Python fallback and focused regression test
without semantic conflict.

The focused Python regression passed (1 test) with this isolated worktree's
`src/` first on `PYTHONPATH`. The closest Rust-owned JSONL artifact regression,
`engine::shard_artifacts::tests::all_empty_shards_leave_empty_jsonl_and_no_csv`,
passed (1 test) with `--features engine --lib` using `sccache` and the dedicated
`/mnt/4tb/aiperf-origin-port-007-target` target directory. The independent
review is recorded as `GRAHAM APPROVED` with no findings in
`.superpowers/sdd/port-origin-007/graham-review.md`.

## Per-commit record: 215be05b6a53

### Upstream intent and Rust comparison

Upstream always projects Baseten `duration_e2e_ms`, `duration_ttft_ms`, and
`cached_tokens_reference` so fidelity outcomes survive loading independently
of KV hints and replay mode. Native already retained E2E privately for
closed-loop delay derivation but dropped TTFT and cached-token reference at its
parse boundary.

### Merge and implementation evidence

`1d20f63c51e7f0e12732d54d61996dc4dc577f71` is a two-parent ours-tree merge
with exact upstream commit `215be05b6a534fb19b84bf83f711db2d20f5bea1` as
its second parent. Its tree equals the reviewed native first-parent tree, so
upstream #39's pending Python content was not imported. Native model, loader,
unit, and real-Parquet public-registry integration work landed in
`ccb8c27c14`; design and exact-diff findings are in
`docs/specs/2026-08-25-native-baseten-outcome-fidelity.md` and
`docs/origin-main-findings/commit-040-215be05b6a.md`.

### Verification and review

TDD recorded the missing `Turn::recorded_outcome` interface first. After the
native projection and Graham fixes, all 13 focused Baseten unit tests and the
one real-Parquet built-in-registry integration test passed using `sccache` and
the dedicated `/mnt/4tb/aiperf-origin-port-040-target`. Runtime all-target
Clippy passed with pre-existing warnings. The complete runtime library run
passed 1,777 tests with one unrelated pre-existing version-snapshot failure;
scoped formatting, docs-current, and range whitespace checks passed. Graham's
first pass found two evidence/test-quality defects, both fixed in `964c3bc32a`;
re-review approved the corrected range with no remaining finding.
