# Native random-pool batch sizes

## Purpose

Define one native contract for authoring and executing file-backed
`random_pool` batch sizes across text, image, audio, and video. The contract
preserves explicit zeroes, validates lossy combinations before execution, and
keeps the loader's independent sampling behavior observable through the real
binary.

## Built

The native dataset pipeline projects all four authored batch sizes through
CLI-only and Config-v2 YAML paths into the random-pool loader. The loader parses
real JSON/JSONL into named pools and modality groups, preserves associated unit
sampling, and selects independent-with-replacement flattened sampling only when
a present modality requests a non-unit size. Resolver and loader guards reject
wrong dataset kinds and formats, lossy identity-bearing batches, directory
non-unit sizes, and empty output turns. Synthetic batching remains independent.

### Authoring and precedence

File-backed `random_pool` accepts the existing public modality controls:

| Modality | CLI | Config-v2 YAML | Loader option |
| --- | --- | --- | --- |
| text | `--batch-size` | `dataset.prompts.batchSize` | `text_batch_size` |
| image | `--image-batch-size` | `dataset.images.batchSize` | `image_batch_size` |
| audio | `--audio-batch-size` | `dataset.audio.batchSize` | `audio_batch_size` |
| video | `--video-batch-size` | `dataset.video.batchSize` | `video_batch_size` |

The existing snake-case aliases remain valid. A value is optional at the
projection boundary so the system distinguishes an omitted flag from an
explicit one, including zero. YAML establishes the base value. Each explicit
CLI flag replaces only its corresponding YAML value. Unset CLI flags preserve
YAML exactly.

These fields are random-pool controls only when resolving a file dataset.
Public datasets reject them. File formats other than `random_pool` reject them
with an error that identifies the effective format and points to
`random_pool`. Synthetic datasets continue using their existing modality batch
controls and are not rejected by file-random-pool validation.

Directory inputs represent multiple named pools. Every explicitly configured
random-pool size must be one for a directory; an authored one is accepted, and
any other value, including zero, fails during config resolution before dataset
loading or endpoint startup.

### Composition

The composer first inventories which modalities have at least one content
item. It enters flattened batch mode if and only if a present modality has a
configured size other than one. A non-unit value for an absent modality is a
no-op and must not erase names or change associated sampling.

In flattened mode the composer samples every present modality independently
with replacement according to its configured size. A modality whose size is
omitted contributes one group; size zero contributes none. The output group's
order remains text, image, audio, then video. Existing deterministic RNG stream
derivation and reproducibility remain unchanged.

Flattened mode refuses inputs whose semantics it cannot preserve:

- more than one named top-level pool;
- any explicitly named modality group;
- image UUID/cache metadata; or
- a configuration which disables every present modality and would emit an
  empty turn.

Unit-mode associated composition remains unchanged, including named pools and
groups. Batch mode intentionally does not preserve cross-modal row association;
users who need association must keep all present modality sizes at one.

### Verification contract

Unit tests exercise real random-pool documents through the public composer,
including all modality counts, zero/default interactions, absent-modality
behavior, and every lossy/empty refusal. Config tests prove CLI-only projection,
YAML preservation, explicit CLI precedence, camel-case aliases, wrong-kind and
wrong-format rejection, directory validation, and synthetic non-interference.
A Rust integration test launches the native `aiperf` binary against a real
file-backed pool and inspects request records to prove the authored size reaches
the loader and wire construction rather than only a parser DTO.

This contract is implemented and independently Graham-approved in the native
port recorded by merge `f1d39ad583f2ed6848b135bf3713240a123a472b`.

## Source anchors

- `rust/runtime/src/config/resolve.rs`
- `rust/runtime/src/dataset/loader/random_pool.rs`
- `rust/cli/src/load.rs`
- `rust/cli/src/yaml.rs`
- `rust/dry-run-tests/tests/`
