<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Origin-main finding 56: random range ratio

## Source finding

Upstream `94fee7338b2cdddf0d69c10526d9f3f81afa64dd` adds two coupled behaviors:

1. a checked uniform ISL/OSL range derived from authored means and a ratio; and
2. `PromptCorpus.RANDOM` generation compatible with vLLM or SGLang's seeded draw order.

The target's only parent is `dd3f09b0c34710470444bad17c9e7050c1cd694a`.
The port must therefore import the target delta only and record the exact target as the
second parent; merging its first-parent history would incorrectly import trackers 40–55.

## Native comparison

Native already accepts `random` as a prompt corpus, filters the tokenizer-provided vLLM
pool, repairs decoded text to the requested token length, and can avoid decode entirely
for raw-token endpoints. That is real prior coverage, but it is not the new behavior:

- Config v2 and the CLI have no random-range-ratio or corpus-style fields.
- The synthetic composer has no ratio distribution or cache sized by conversation count.
- Random prompt tokens are independent RNG choices. They do not use upstream's
  `allowed[(offset + request_index + token_index) % len]` sequence.
- Lengths and prompt offsets use separate derived AIPerf streams instead of the reference
  all-ISL, all-OSL, all-offset sequence.
- There is no SGLang full-vocabulary/legacy-MT19937 mode.

This is a native product gap: the same authored seed and parameters produce a different
request population and different prompt bodies.

## Test mapping

The Python suite is mapped to native Rust behavior tests as follows:

| Upstream surface | Native evidence |
| --- | --- |
| ratio parsing/validation and exclusivity | CLI, YAML, protocol-v2, and dataset-build unit tests |
| vLLM/SGLang inclusive bounds and special-token adjustment | checked range-policy unit tests with pinned vectors |
| all ISLs then OSLs then offsets | one seeded-plan golden-vector test per style |
| random pool selection and request-index arithmetic | prompt-generator unit tests for special exclusion/full vocab and cache fallback |
| decode/encode exact-length repair | existing repair tests plus style-backed regression tests |
| prefix and zero-body guards | synthetic composer tests for zero without prefix and additive prefix accounting |
| real request population | Rust E2E against the in-repo mock server, including seed reproducibility |

The native port deliberately keeps count/hash-backed trace synthesis on its existing
AIPerf deterministic corpus contract. The reference range-ratio stream is a synthetic
dataset contract and is not silently applied to recorded trace hashes.

## Implemented native port

The target delta is retained as exact second-parent ancestry by merge `cd31c0ae5a`.
Native implementation commits through `3af08f8807` provide:

- checked vLLM/PCG64 and SGLang/RandomState-MT19937 range plans;
- exact all-ISL→all-OSL→offset draw order, wide-seed XOR folding, continuation draws,
  request-index arithmetic, style-specific token pools, and bounded failures;
- Config-v2, YAML, CLI, and protocol-v2 projection;
- token-additive prefix composition, prefix-only rescue, raw-ID composition, exact text
  repair, and non-range special-token ISL compensation; and
- a real Python/native profile A/B harness over the production HTTP request path.

The earlier Baseten prerequisite commit `b244e54ea4` exists only because this branch is
based on old shared HEAD `b009535bb8`. Integration must retain the current shared Baseten
outcome work from `428e820510` and the open-loop correction from `aaba0e6990`; it must not
reapply `b244e54ea4` over those newer semantics or restore closed-loop-only outcomes.

## Mandatory semantic audits

- [Audit 1: RNG and reference stream](commit-056-audit-1-rng-reference-stream.md)
- [Audit 2: config, dataset, formatter, and prefix semantics](commit-056-audit-2-config-dataset-prefix.md)
- [Audit 3: Python-to-native production E2E](commit-056-audit-3-production-e2e.md)

All three audits compare exact upstream `94fee7338b` with implementation tree
`3af08f8807`, cite source lines and executable evidence, and report no unresolved
divergence after the rebuilt production gate passed 13/13 tests and all 48 ordered
captures.

## Verification receipt before independent review

- Runtime engine random-range filter: 6 passed, 0 failed.
- Native range integration: 3 passed, 0 failed after one recorded RED caused by an
  obsolete whole-plan equality assertion.
- Python-generated length/offset/token vectors: 1 passed, 0 failed; generator `--check`
  and Ruff both passed.
- CLI ratio/style: 2 passed, 0 failed; prompt-corpus surfaces: 7 passed, 0 failed.
- Non-range special-token compensation: 1 passed, 0 failed.
- Full production Python/native A/B: 13 passed, 0 failed; six cases × eight requests =
  48 ordered method/route/content-type/body/token-ID matches.

Independent Graham approval and campaign closure remain pending; these receipts do not
authorize provenance or tracker closure by themselves.
