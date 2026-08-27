<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark security-scope course correction

Status: normative amendment to the native streaming dataset and shadow-replay
design and implementation plans.

This amendment narrows generation-one security work to the boundaries a load
generator actually crosses. AIPerf is a benchmarking and replay system. Its
primary obligations are reliable continuation, truthful restart, bounded
resource ownership, reproducibility, throughput, and useful partial results.
It is not a general secret-storage or hostile-multitenant execution platform.

Where this amendment conflicts with the base design or any native-streaming
implementation plan, this amendment wins.

## Decision

Generation one ships recorded-input replay. Resumable `target_closed_loop`
state, encrypted checkpoint participants, external key resolution,
XChaCha20-Poly1305 envelopes, key-loss behavior, and zeroization are deferred.
An implementation may support ephemeral target-closed-loop execution only when
it does not claim restartability for target-derived state.

The base `streaming` feature owns no direct encryption or secret-erasure
dependency. Its feature graph is `streaming = ["engine"]`. S3 remains isolated
behind `streaming-s3` and uses the provider SDK's existing authentication and
TLS support.

## Invariants

### S1 — Reliability is the governing priority

An ordinary source, decode, session, action, checkpoint-attempt, or result
export failure is handled through the typed reliability policy. Security
machinery must not turn an otherwise recoverable benchmark fault into a run
failure unless the selected run crosses a concrete external trust boundary
whose failure makes the requested operation impossible or untruthful.

### S2 — Hashing proves identity and integrity, not trust

BLAKE3 remains the single deterministic content hash for stable logical
identity, immutable object and segment corruption detection, canonical conflict
detection, CAS lineage, idempotent retry, and restart verification. These
digests are not signatures, credentials, or proof of an adversarial peer.

No layer recomputes a digest merely to establish a second security vocabulary.
At an ownership boundary, the producer computes the canonical digest once and
the consumer verifies it once; verified typed authority carries that fact
through later seams.

### S3 — No generation-one encrypted session-state substrate

Generation-one checkpoint schemas and capability agreement do not advertise,
require, or refuse an `encrypted_sensitive_session_state` capability. They do
not define key IDs, key resolvers, nonce allocation, authenticated-encryption
AAD, key rotation, key loss, or secret zeroization.

Recorded source inputs remain restartable through ordinary checkpoint objects.
Target-derived closed-loop state is either ephemeral and explicitly
non-restartable or outside generation one.

### S4 — Real external boundaries reuse existing mechanisms

HF and S3 access retain provider credentials, SDK TLS, refresh, and redacted
diagnostics. Cross-host cellular control retains the existing process admission
and replay-ledger mechanisms. Streaming adds no parallel key hierarchy, custom
signing layer, payload-encryption system, or bespoke credential protocol.

No-follow acquisition, private scratch, bounded decompression, strict DTOs,
length checks, and secret-absence assertions remain because they directly
support correctness, boundedness, and trustworthy benchmark artifacts.

### S5 — Ownership handoff is a correctness transaction

Streaming placement describes cell movement as a fenced ownership-handoff
generation, not as an authenticated state-transfer transaction. Correctness is
defined by ownership epoch, drain, checkpoint commit, and old-owner release.
Transport admission is an independent property supplied by the existing
cellular boundary.

### S6 — Results survive ordinary failure

Checkpoint and result dumping continue to prioritize atomic publication,
recoverable generations, bounded object retention, and the ability to publish
useful partial results. Encryption or key availability is not a prerequisite
for result compaction, object leases, bounded garbage collection, or ordinary
restart.

## Executable plan corrections

1. Remove direct `chacha20poly1305` and `zeroize` workspace/runtime dependency
   wiring and regenerate the lockfile. Transitive cryptography used by TLS or
   provider SDKs may remain.
2. Remove sensitive-state encryption from Task 2 capability agreement and from
   generation-one protocol/startup refusal matrices.
3. Defer pipeline Task P6 in full. Tasks P1-P5 and P7 remain independent of it.
4. Make checkpoint/results Task 5F3 depend on 5D and 5F2 only; it owns object
   leases and bounded garbage collection, not encrypted state.
5. Remove encrypted restore, wrong-key, key-loss, and zeroization cases from the
   generation-one product verification matrix.
6. In cellular work, reuse the existing admission authority and reuse the
   verified canonical payload digest after one sender computation and one
   receiver verification.

## Explicit non-goals

- protecting benchmark checkpoints from an attacker who can already modify the
  configured result backend;
- confidentiality for target-derived session state;
- streaming-specific signatures, certificates, keys, or rotation;
- treating content digests as authentication;
- failing a benchmark because an unselected encryption capability is absent.

If a future product requirement introduces a real confidentiality boundary,
it must arrive as a separate, non-default feature with its own threat model and
backend owner. It must not silently expand `streaming` or `streaming-s3`.

