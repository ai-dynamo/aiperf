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
| 3 | `34b2be2ee1` | SGLang speculative metrics | merged | applicable | Native speed-bench recognition is not profile console-exporter parity. |
| 4 | `6db948524e` | SageMaker malformed timestamp | merged | already-covered | Native loader returns validation error. |
| 5 | `ce715ae849` | agentic think-time idle guard | complete | applicable | Merged in `6e8da730e8`; native scenario default/lock port in `ad4c2a54f4`, `6e78b41d35`, and `729c59f64a`; Graham approved. |
| 6 | `93b6223373` | telemetry field rename | complete | applicable | Merged in `4ab850c79d`; native artifact migration and custom-prefix closure in `726fcb614b`, `4763b53f0f`, and `978bc93fea`; Graham approved. |
| 7 | `86ea3f7deb` | detailed JSONL fallback | pending | already-covered | Native aggregation uses `profile_export.jsonl`; separately sync independent Python profile. |
| 8 | `5566aae1e1` | ShareGPT batch encoding | pending | applicable | Native composition has no batch-tokenizer seam. |
| 9 | `844efe1b36` | full synthesis prefix blocks | pending | already-covered | Native synthesis preserves block identity. |
| 10 | `ffc943a9fe` | FFmpeg CVE bump | pending | shared-product-action | Shared shipped Dockerfile remains on 8.1.1; security port required. |
| 11 | `d55ae21d34` | ShareGPT tutorial timeout | pending | not-applicable | Documentation only. |
| 12 | `8148999496` | LCB trust_remote_code | pending | already-covered | Rust-launched Python uses a script-free pinned loader; tokenizer policy is not the relevant evidence. |
| 13 | `d32f4bb98e` | canonical spec-decode metrics | pending | applicable | Generic token fields exist; full canonical/vLLM model does not. |
| 14 | `4fe3ff7154` | cache-bust enables wrapping | pending | already-covered | Native graph wrap policy has this rule. |
| 15 | `2eb04aa2f8` | MMVU docs memory | pending | not-applicable | Documentation only. |
| 16 | `5ad08166a1` | profiling grace default | pending | applicable | Omitted duration-based grace does not materialize upstream 30-second default. |
| 17 | `c02d02db28` | idle cap after barrier | pending | already-covered | Native continuation guard handles pending tasks. |
| 18 | `2f413f0dec` | minified schemas | pending | shared-product-action | Python schemas ship in the unified wheel; scheduler half has no native port. |
| 19 | `9b5f5f7282` | warmup prefix stand-down | pending | superseded | Reassess together with #32 warmup-isolation targets. |
| 20 | `446d2cd4b3` | empty raw_messages delta | pending | applicable | Rust currently synthesizes a turn; upstream treats `[]` as zero messages. |
| 21 | `20eb25626a` | docstring policy | pending | not-applicable | Contributor docs only. |
| 22 | `1d1829540b` | structured cache-bust | pending | already-covered | Native marker ownership is equivalent. |
| 23 | `fc7bbf3bdd` | DAG branches | pending | already-covered | Graph-IR is the replacement architecture; needs behavior-level regression only. |
| 24 | `f8c8e36533` | BranchStats test | pending | not-applicable | Test-only Python correction. |
| 25 | `5a24e0d7c7` | average error-rate SLA | pending | applicable | Search still projects p99/unscaled fractional threshold; pair with #48. |
| 26 | `093038afab` | JSON-array endpoint responses | pending | already-covered | Native typed array parsers cover the contract. |
| 27 | `f0128e105b` | reset/profiler hooks | pending | applicable | Hooks exist; partial-start rollback and aggregate stop failures need parity decision. |
| 28 | `aaaa72e69d` | arrival tutorial flag | pending | not-applicable | Documentation only. |
| 29 | `c2f5e9d459` | Python enum cache | pending | not-applicable | Python-specific hot path. |
| 30 | `9be59a9636` | AMD telemetry tests | pending | already-covered | Production source exists; CLI E2E parity remains a coverage improvement. |
| 31 | `59cd43e43a` | Responses mock model | pending | applicable | Route/model exist but omit `min_tokens`, `ignore_eos`, and typed-request recording. |
| 32 | `1b34de0637` | warmup-isolation cache-bust | pending | applicable | Native cannot represent phase-gated targets. |
| 33 | `37467dc38c` | force content parts | pending | applicable | Add bootstrap-captured endpoint policy. |
| 34 | `00ba1c5db3` | audio transcription | pending | applicable | New multipart endpoint, parser, and mock route. |
| 35 | `03c9c6ddc5` | LCB grading concurrency | pending | applicable | Delegation/detail are present; upstream multiplexing and unconditional group reaping remain open. |
| 36 | `6480e5467f` | synthesis YAML overrides | pending | applicable | Flag-only support exists; CLI-over-YAML overlay is absent. |
| 37 | `23ed221c3d` | CI speed | pending | not-applicable | CI-only work. |
| 38 | `c26fe88bd8` | AgentX FAQ | pending | not-applicable | Documentation only. |
| 39 | `1e32a51318` | Baseten load performance | analysing | unassessed | Requires benchmark before claiming a port gap. |
| 40 | `215be05b6a` | Baseten outcome fidelity | pending | applicable | Only `duration_ttft_ms` and `cached_tokens_reference` remain missing. |
| 41 | `516faa12c8` | CODEOWNERS | pending | not-applicable | Repository metadata. |
| 42 | `ce453582c7` | CONTRIBUTING spelling | pending | not-applicable | Documentation only. |
| 43 | `6ed4823d12` | cache-bust help link | pending | not-applicable | Documentation only. |
| 44 | `082a51827e` | TraceLab dataset | pending | applicable | No native loader/config path. |
| 45 | `21f8ad7b3e` | high-resolution pacing | analysing | unassessed | timerfd exists; late-slot behavior needs characterization. |
| 46 | `e659d2a95a` | FFmpeg codec allowlist | pending | shared-product-action | Audit native codec defaults before porting shared Dockerfile policy. |
| 47 | `9e96b499d1` | aiohttp minimum | pending | already-covered | Bundled Python bound already satisfies it. |
| 48 | `260d00f5e9` | adaptive error-rate units | pending | applicable | Adaptive SLA differs from exported percentage/cancellation contract; pair with #25. |
| 49 | `88242293b5` | verbatim system prompts | pending | applicable | Add CLI/file projection to existing composition seam. |
| 50 | `ade1f69eb1` | seamless phase transitions | pending | already-covered | Native lowers incoming flag to predecessor handoff. |
| 51 | `324bb05773` | per-chunk usage | pending | applicable | No first-chunk token multiplicity or option. |
| 52 | `810fd8bdd4` | vLLM spec-decode wire | pending | applicable | Root metrics/histogram/trailing usage absent. |
| 53 | `e5ebe915df` | enum inequality | pending | not-applicable | Python enum behavior only. |
| 54 | `bfe33151de` | rejected peak diagnostics | pending | applicable | Actual target is WEKA/Dynamo selection, not Baseten. |
| 55 | `dd3f09b0c3` | session-affinity header | pending | applicable | Add default-on `X-Session-Affinity`, not opt-in `X-Session-ID`. |
| 56 | `94fee7338b` | random range ratio | pending | applicable | Random corpus exists; ratio surface/sampling is absent. |
| 57 | `e10d53b1d3` | version 0.13.0 | pending | shared-product-action | Coordinated release decision, not standalone Rust port. |
| 58 | `9b60a3d479` | FakeTokenizer | pending | not-applicable | Python test harness only. |
| 59 | `c9288da6c1` | mmap conversation cache | pending | not-applicable | No equivalent native mmap backend. |
| 60 | `c2889280a6` | random_pool batch sizes | pending | applicable | Runtime supports modalities but projection exposes only image. |

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
