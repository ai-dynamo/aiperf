<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Publishing Traces for AIPerf

Upload a trace in any format AIPerf already reads, add a small `aiperf.json`, and anyone
can run it with `--input-file hf://<org>/<repo>` (or `s3://...`) with no converter and no
AIPerf change. To run a published trace, see [Run a Published Trace](remote-datasets.md).

---

## Recommended Format

`mooncake_trace` JSONL (optionally gzipped) is the most general: each line carries a
`timestamp` plus either `input_length`/`hash_ids` (anonymized), `text_input`, `messages`,
or a verbatim `payload`. See [Trace Replay with Mooncake Traces](../benchmark-modes/trace-replay.md)
for the record format. Weka, Bailian, BurstGPT, Baseten, TraceLab, and SageMaker Data
Capture formats are also supported; list every loader name with
`aiperf plugins custom-dataset-loader`.

---

## Where to Put `aiperf.json`

| You publish | Manifest | Consumers run |
|---|---|---|
| A repo, a folder, or an S3 prefix | `aiperf.json` at its root (required) | `hf://<org>/<repo>`, `hf://<org>/<repo>/<folder>`, `s3://<bucket>/<prefix>/` |
| One file | `<file>.aiperf.json` next to it (optional, without `path`) | `hf://<org>/<repo>/<file>`, `s3://<bucket>/<key>`, `https://.../<file>` |

A root manifest names the data file or folder with `path` and its loader with `format`. A
sidecar only adds defaults and requirements for a file the consumer already pointed at, so it
must not set `path`. A repo, folder, or prefix without `aiperf.json` cannot be run.

HTTPS references support single files and sidecars only. The sidecar is looked up at
`<url>.aiperf.json`, and not at all when the URL has a query string (a presigned URL signs
one object). A server that answers the sidecar path with an HTML page is treated as having no
sidecar.

A folder can hold several files (for example a `random_pool` or `raw_payload` directory).
Only folders whose format reads a directory work as `path`, with one exception: a folder of
parquet shards works for any format that reads parquet. It follows the `train` split rule in
[Parquet Datasets](remote-datasets.md#parquet-datasets): a folder with several splits and none
named `train` fails the access check, so publish one split or name it `train`.

---

## `aiperf.json`

```json
{
  "aiperf_manifest_version": 1,
  "format": "mooncake_trace",
  "path": "data/chat-2026-08.jsonl.gz",
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "block_size": 512,
  "requires": {"aiperf": ">=0.13", "endpoint_types": ["chat"], "max_context_tokens": 131072},
  "description": "Anonymized chat traffic, Aug 2026",
  "license": "cc-by-4.0",
  "tokenizer": "Qwen/Qwen3-32B"
}
```

| Field | Required | Meaning |
|---|---|---|
| `aiperf_manifest_version` | yes | Currently `1`. A newer version makes older AIPerf releases fail with a request to upgrade. |
| `format` | root manifests | A loader name from `aiperf plugins custom-dataset-loader`. A consumer's `--custom-dataset-type` must match it. |
| `path` | root manifests | One file or one folder, relative to the root. No `..`, no absolute paths, no backslashes. |
| `sha256` | no | Lowercase hex digest, checked after download. Single files only; ignored for folders. |
| `block_size` | no | Hash block size the trace was recorded with. A `--prompt-input-tokens-block-size` (`--isl-block-size`) on the command line still wins. |
| `requires.aiperf` | no | A [PEP 440](https://peps.python.org/pep-0440/) version specifier, such as `>=0.13`. |
| `requires.endpoint_types` | no | Allowed `--endpoint-type` values. Other endpoint types fail before the download. |
| `requires.max_context_tokens` | no | Peak input+output tokens of any request. AIPerf warns if it exceeds `--max-context-length`. |
| `description`, `license`, `tokenizer` | no | Informational. |

Notes:

- The `sha256` above is a placeholder of 64 zeros: replace it with the file's real digest, from
  `sha256sum data/chat-2026-08.jsonl.gz` (`shasum -a 256` on macOS). A mismatch fails the run
  with a sha256 mismatch error.
- `requires.aiperf` is checked against the installed AIPerf version. An install from source
  reports the version in `pyproject.toml`, which stays at the last release until the next
  release bumps it, so require the oldest release your dataset works with (`>=0.13` above),
  not a release that is still in progress. Pre-release builds sort below their release under
  PEP 440: `0.14.0.dev12` does not satisfy `>=0.14`.
- The `max_context_tokens` check is a warning only, and it only runs for formats that accept
  `--max-context-length`, which today is Weka. For other formats the field is informational.
- Unknown fields, including unknown `requires` keys, are ignored with a warning, so newer
  manifests still load on older AIPerf.
- A manifest or sidecar larger than 1 MiB is rejected.
- If the manifest sets `format` or `block_size`, AIPerf applies it as if the consumer had passed
  it. If it conflicts with the consumer's own options, the run fails with an error that names
  the conflict.
- AIPerf checks the downloaded file's first record against the declared `format` and fails if it
  does not look like that format, so test the manifest against the real file before announcing it.

---

## Publishing to Hugging Face

```bash
hf upload <org>/<repo> data/chat-2026-08.jsonl.gz data/chat-2026-08.jsonl.gz --repo-type dataset
hf upload <org>/<repo> aiperf.json aiperf.json --repo-type dataset
```

AIPerf pins the repo's commit SHA on every run, so the run log always names the exact version
it replayed. Tell consumers to pin with `@<commit>` or a tag (`hf://<org>/<repo>@v1`) when they
need comparable results across runs. Gated repos work; consumers must accept the terms on the
dataset page and authenticate.

## Publishing with `datasets.push_to_hub`

A dataset pushed with `datasets.push_to_hub` already runs, with no manifest:

```python
from datasets import Dataset

Dataset.from_list(rows).push_to_hub("<org>/<repo>", config_name="mooncake")
```

Consumers run `--input-file hf://datasets/<org>/<repo>/mooncake --custom-dataset-type mooncake_trace`;
see [Parquet Datasets](remote-datasets.md#parquet-datasets) for how AIPerf finds the shards.
Adding an `aiperf.json` still buys the declared `format` (so consumers can drop
`--custom-dataset-type`), the `requires` checks, and `sha256` verification, which covers single
files only.

## Publishing to S3

```bash
aws s3 cp data/chat-2026-08.jsonl.gz s3://my-traces/chat/data/chat-2026-08.jsonl.gz
aws s3 cp aiperf.json s3://my-traces/chat/aiperf.json
```

Consumers run `--input-file s3://my-traces/chat/` and need `uv pip install 'aiperf[aws]'` plus
read access (`s3:GetObject`, `s3:ListBucket` for a prefix, and `s3:GetObjectVersion` for
`?versionId=`). Enable bucket versioning so
consumers can pin one object with `?versionId=`. For a prefix, AIPerf pins a digest of the
manifest and the listing, re-lists before downloading, and fetches each object conditionally on
its listed ETag, so an object replaced mid-run fails the run instead of mixing versions. Every
key under the prefix must be storable as a local file: a key containing `..`, `.`, an empty
segment, or a backslash fails the run, naming the key, and so does a colon or a drive letter
when AIPerf runs on Windows. Folder-marker keys ending in `/` are ignored.

## Publishing over HTTPS

Host the file and, optionally, `<file>.aiperf.json` beside it on any server. Presigned S3 or
GCS URLs work too (AIPerf falls back from `HEAD` to a ranged `GET` when the server refuses
`HEAD`), but a presigned URL has no sidecar, so consumers pass `--custom-dataset-type`
themselves. A server should send an `ETag` (or `Last-Modified`) so consumers can cache the
download; without one, AIPerf downloads the file again on every run.
