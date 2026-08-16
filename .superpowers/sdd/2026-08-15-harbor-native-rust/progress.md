# SDD ledger — plan: docs/plans/2026-08-15-harbor-native-rust.md

## Pre-flight interface scan

| Tasks / interface | Producer → consumer | Finding / ruling |
| --- | --- | --- |
| 1 → 2 | immutable IDs, source manifests → importer | Implemented prior to this session; importer consumes only stable public eval identities. |
| 1 → 3 → 4 | trial/attempt/evidence → sandbox and verifier/regrade | Task 4 must add only append-only regrade records; it must not mutate `ScoreVersion` or `EvidenceEvent`. |
| 4 → 6 | declared artifact transfer, verifier modes, score versions → P0 composition | Product tests must exercise actual public contracts, not mock-call assertions. |
| 5 → 6 | semantic lowerer and paired baseline → P0 paired report | Current semantic surface needs a report type before P0 acceptance can prove deltas. |
| 1 → 7 | immutable task/trial/evidence → P1/P2 types | Extension contracts must validate only and must not add online registry or provider execution. |
| 6 → docs | implementation behavior → design records/index | Documentation must be updated only after P0 composition exists and claims must be source-verified. |

Ruling: Existing Task 1 omitted the planned `TaskSpec`, but later code currently exposes only the smaller imported-task vocabulary. I will add required public immutable contracts incrementally where consumed by Tasks 4, 6, and 7 rather than retrofit speculative fields wholesale. This costs a later compatibility change if P0 composition reveals a missing immutable identity field.

Ruling: Task 4 is incomplete despite existing reward and artifact types because no `RegradeRequest` or regrade API exists. Implement it before Task 7/P0 so P0 tests have a real append-only score operation. This costs little surface churn and satisfies the plan's explicit API.

Task 1: complete (commits 74f00be35e..b8e33bc544, prior focused review not available after agent loss)
Task 2: complete (commits 26fa9afa93..f48cdde77d, prior focused review not available after agent loss)
Task 3: complete (commits a2ca66a673..30c4f27c07, prior focused review not available after agent loss)
Task 4: in progress (existing verifier/reward artifacts and score contracts at e0d69ed394; regrade API pending)
Task 5: complete (commits a4e2c3ba70..02442e9415, prior focused review not available after agent loss)
Task 6: pending
Task 7: pending
Task 4: fix round 1/5 (artifact path and sandbox API added; review found malformed-reward recording and strict DTO decode still open; commits fa1cfff1e0..d6496c8592)
Task 4: fix round 2/5 (identity decode and automatic malformed-reward evidence added; strict score/reward DTO decode remains open; commits 6e38bb1167..c5e9c70c6f)
Task 4: complete (strict score/reward/verifier/regrade decoding, canonical artifact paths, declared-only verifier handoff, and append-only regrade covered by focused contracts; commits f5ccf58c11..e217a5a663)
Task 7: complete (provider capability refusal, offline-safe registry references, immutable trajectory manifests, and evidence-backed task health contracts; commits 6aad3cde6c..0310b6f079)
Task 6: in progress (native P0 lifecycle composition and fixtures landed in a063408a7c..e00e01a823: local installed/external contracts, pinned-Git identity after `HEAD` mutation, declared-artifact isolation, score/regrade lineage, and paired comparison. It remains open until the prescribed P0 acceptance command, Docker/security coverage, documentation checks, and full branch review have all passed.)
Task 6: Docker verifier-isolation acceptance evidence added in bf5f0613a6: `AIPERF_E2E_BIN=$PWD/target/debug/aiperf cargo test -p aiperf-e2e-tests --test test_harbor_lifecycle_e2e -- --ignored --exact pinned_docker_lifecycle_withholds_agent_credential_and_workspace_and_never_invokes_harbor` passed 1/1 in 7.01s. The real Docker path proves the agent receives its configured credential while the fresh verifier sees neither that credential nor an undeclared agent workspace file, and an injected `harbor` process spy remains uninvoked. Task 6 remains open for the full prescribed acceptance suite, documentation checks, and final branch review.
