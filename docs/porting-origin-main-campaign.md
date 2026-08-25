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
| 5 | `ce715ae849` | agentic think-time idle guard | pending | already-covered | Source comparison supports equivalent native behavior. |
| 6 | `93b6223373` | telemetry field rename | pending | applicable | Native artifact still emits `dcgm_url`; requires contract migration decision. |
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
