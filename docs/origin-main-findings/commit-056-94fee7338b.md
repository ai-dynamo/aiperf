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
