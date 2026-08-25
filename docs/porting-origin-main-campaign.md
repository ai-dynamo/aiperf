# origin/main Rust Porting Campaign Ledger

## Scope

This ledger tracks every commit reachable from `origin/main` and not reachable
from `ajc/rust-merge-main` at campaign start (`0b67e08284ef`). The baseline
comparison was `origin/main` at `c2889280a66f`; `git rev-list
HEAD..origin/main` reported 60 commits.

For each commit, the campaign records the upstream intent, the Rust parity
assessment, an actual non-fast-forward merge commit, any Rust-port design and
Sol plan, TDD evidence, verification, and the required Graham review.

## Status vocabulary

- `pending`: not yet inspected.
- `analysing`: upstream and Rust code comparison in progress.
- `merged`: an actual merge commit has incorporated the upstream commit.
- `rust-porting`: a Rust delta was identified and is being specified/planned.
- `complete`: merge, applicable Rust port, tests, and Graham review are all
  evidenced.
- `not-applicable`: no Rust behavior corresponds to the upstream change;
  evidence is recorded.

## Campaign inventory

| Order | Upstream commit | Subject | Status | Evidence |
| --- | --- | --- | --- | --- |
| 1 | `817a8d84ddb9` | fix(accuracy): grade LCB codegen in an out-of-process worker (#1145) (#1175) | rust-porting | Merge commit `1c03271dac3e` has upstream as its second parent. Focused upstream tests: 35 passed, 1 skipped. The native evaluator's LCB batch path still forks via `asyncio.to_thread`; the port spec is `docs/specs/2026-08-25-native-lcb-codegen-worker.md`. |
+| 2 | `0883bd1aee` | chore: bump aiperf version to 0.12.0 (#1194) | pending | — |
| 3 | `34b2be2ee1` | feat: support Speculative Decoding metrics in AIPerf (#1153) | pending | — |
| 4 | `6db948524e` | fix: uncaught valueerror in sagemaker loader (#1199) | pending | — |
| 5 | `ce715ae849` | fix(agentic): preserve trace think time with global idle guard (#1201) | pending | — |
| 6 | `93b6223373` | refactor(telemetry): rename TelemetryRecord.dcgm_url → telemetry_source_url (#1200) | pending | — |
| 7 | `86ea3f7deb` | fix(cli-runner): fix multi-run detailed aggregation JSONL fallback (#1203) | pending | — |
| 8 | `5566aae1e1` | perf(dataset): batch-encode ShareGPT to fix 300s configuration timeout (#1206) | pending | — |
| 9 | `844efe1b36` | fix(synthesize): keep shared prefix blocks full so synthesized traces replay (#1208) | pending | — |
| 10 | `ffc943a9fe` | fix(security): bump ffmpeg to 8.1.2 for CVE-2026-8461 (#1215) | pending | — |
| 11 | `d55ae21d34` | fix(docs): raise dataset timeouts for cold-cache ShareGPT tutorial run (#1224) | pending | — |
| 12 | `8148999496` | fix(accuracy): remove trust_remote_code from lcb dataset load (#1213) | pending | — |
| 13 | `d32f4bb98e` | feat(spec-decode): canonical per-request acceptance metrics (#1191) | pending | — |
| 14 | `4fe3ff7154` | fix(timing): let an active cache-bust target satisfy the dataset-wrap opt-in (#1225) | pending | — |
| 15 | `2eb04aa2f8` | fix(docs): bound MMVU video memory so docs E2E stops OOMing (#1223) | pending | — |
| 16 | `5ad08166a1` | fix(config): preserve default profiling grace period (#1230) | pending | — |
| 17 | `c02d02db28` | fix(agentic): enforce idle cap after barrier deferral (#1205) | pending | — |
| 18 | `2f413f0dec` | refactor(schemas): minify generated JSON schemas (#1210) | pending | — |
| 19 | `9b5f5f7282` | fix(warmup): stand down warmup prefix when cache-bust is active (#1228) | pending | — |
| 20 | `446d2cd4b3` | fix(endpoints): treat empty raw_messages as a no-op delta, not a synth turn (#1220) | pending | — |
| 21 | `20eb25626a` | docs(standards): allow multi-line docstrings in project code (#1234) | pending | — |
| 22 | `1d1829540b` | feat(cache-bust): support all structured workloads (#1233) | pending | — |
| 23 | `fc7bbf3bdd` | feat(dag): per-round authored branches for the orchestrator spine (#1218) | pending | — |
| 24 | `f8c8e36533` | fix(dag): update BranchStats test expectations for graphs_admitted/graphs_completed_to_end (#1238) | pending | — |
| 25 | `5a24e0d7c7` | fix(search): evaluate error-rate SLA on average percentage (#1155) | pending | — |
| 26 | `093038afab` | fix(endpoints): handle top-level JSON array responses (#1195) | pending | — |
| 27 | `f0128e105b` | feat: add endpoint reset_kv_cache and server_profiler control hooks (#1163) | pending | — |
| 28 | `aaaa72e69d` | fix(docs): correct output directory flag in arrival-patterns tutorial (#1239) | pending | — |
| 29 | `c2f5e9d459` | perf: cache normalized value/hash on hot-path string enums (#1184) | pending | — |
| 30 | `9be59a9636` | test(amdsmi): add e2e integration tests for AMD mock telemetry path (#1192) | pending | — |
| 31 | `59cd43e43a` | feat(mock-server): add ResponsesRequest model with full dispatch plumbing (#1000) | pending | — |
| 32 | `1b34de0637` | fix(cache-bust): add WARMUP_ISOLATION_* targets; remove unconditional warmup prefix (#1256) | pending | — |
| 33 | `37467dc38c` | feat(endpoint): add AIPERF_ENDPOINT_FORCE_CONTENT_PARTS to control content serialization (#1259) | pending | — |
| 34 | `00ba1c5db3` | feat(endpoints): add audio_transcription endpoint type (#1247) | pending | — |
| 35 | `03c9c6ddc5` | feat(accuracy): restore codegen grade concurrency (AIP-1094) (#1237) | pending | — |
| 36 | `6480e5467f` | fix(config): apply synthesis CLI overrides to YAML (#1258) | pending | — |
| 37 | `23ed221c3d` | perf(ci): reduce test wall-clock time across platforms (#1261) | pending | — |
| 38 | `c26fe88bd8` | fix(docs): Remove AgentX FAQ (#1266) | pending | — |
| 39 | `1e32a51318` | perf: speed up Baseten trace loading (#1248) | pending | — |
| 40 | `215be05b6a` | fix: always project recorded-outcome columns so fidelity survives load (#1269) | pending | — |
| 41 | `516faa12c8` | chore: Update CODEOWNERS to aiperf-codeowners team (#1271) | pending | — |
| 42 | `ce453582c7` | docs: spell out Visual Studio Code (#1272) | pending | — |
| 43 | `6ed4823d12` | docs(cache-bust): add reference link to cache-bust.md in --cache-bust description (#1264) | pending | — |
| 44 | `082a51827e` | feat(dataset): add tracelab dataset support (#1262) | pending | — |
| 45 | `21f8ad7b3e` | perf(timing): high-resolution rate-loop pacing for exact rate delivery (#1185) | pending | — |
| 46 | `e659d2a95a` | feat: build ffmpeg with a minimal codec allowlist (#1232) | pending | — |
| 47 | `9e96b499d1` | chore: raise aiohttp minimum to 3.14.3 (#1277) | pending | — |
| 48 | `260d00f5e9` | fix: align adaptive-scale request_error_rate SLA with exported metric unit (#1240) | pending | — |
| 49 | `88242293b5` | feat(dataset): add --system-prompt/--system-prompt-file for verbatim system prompts (#1268) | pending | — |
| 50 | `ade1f69eb1` | fix(timing): honor incoming seamless phase transitions (#1263) | pending | — |
| 51 | `324bb05773` | feat(metrics): add --per-chunk-usage to fix TPS/user inflation from bundled first streamed chunk (#1279) | pending | — |
| 52 | `810fd8bdd4` | fix(spec-decode): align vLLM adapter with #48915 review-updated wire format (#1282) | pending | — |
| 53 | `e5ebe915df` | fix(enums): make != the negation of == on case-insensitive str enums (#1314) | pending | — |
| 54 | `bfe33151de` | fix(dataset): report smallest peak context when all traces are rejected (#1287) | pending | — |
| 55 | `dd3f09b0c3` | feat(http): send session affinity header by default (#1312) | pending | — |
| 56 | `94fee7338b` | feat(dataset): random range ratio + PromptCorpus.RANDOM aligned with vLLM bench (#850) | pending | — |
| 57 | `e10d53b1d3` | chore: bump aiperf version to 0.13.0 (#1317) | pending | — |
| 58 | `9b60a3d479` | fix(test-harness): implement num_prompt_special_tokens on FakeTokenizer (#1322) | pending | — |
| 59 | `c9288da6c1` | fix(mmap): make get_conversation thread-safe, prefault pages, drop executor hop (#1245) | pending | — |
| 60 | `c2889280a6` | fix(dataset): restore random_pool batch-size support with --input-file (#1274) | pending | — |

## Per-commit record: 817a8d84ddb9

### Upstream intent

Move LiveCodeBench code-generation grading to a supervised child process so
potentially wedged grading does not hold the parent benchmark process.

### Initial Rust comparison

Rust has a versioned evaluator-worker protocol with `load`, `next_problems`,
`grade_batch`, and `shutdown`, plus subprocess supervision in
`rust/runtime/src/accuracy_core/worker.rs`. The remaining inspection will
compare lifecycle, failure propagation, reaping, and LiveCodeBench-specific
selection before deciding whether any Rust delta remains.

### Merge evidence

`1c03271dac3eb6465538dabf6950fd255baeac7d` is a two-parent merge with
`817a8d84ddb90d1e12c2a03327e16d853bb4e6e0` as the second parent. The focused
upstream suite produced `35 passed, 1 skipped`; pytest emitted cleanup warnings
for pre-existing Docker-owned temporary paths. Project-wide commit hooks then
reported pre-existing unrelated import and ergonomics/ruff-baseline failures;
the staged merge touched none of their listed source paths.

### Port decision

Applicable. Rust's `PythonEvaluator` starts `aiperf.accuracy.worker`, whose
`AccuracyWorker._grade_lcb_batch` still invokes `_run_codegen_metrics` through
`asyncio.to_thread`. That retains the upstream fork-from-thread hazard on the
native Rust evaluation route. The selected correction reuses the merged
`CodegenGradingWorker` inside that evaluator, recorded in
`docs/specs/2026-08-25-native-lcb-codegen-worker.md`.

### Required evidence before close

- Upstream semantic diff and current Rust counterpart analysis.
- A true merge commit containing the upstream ancestry.
- A feature spec and Sol-produced plan if any Rust delta is applicable.
- TDD red/green evidence and all applicable tests.
- A full Graham review with every finding resolved.
