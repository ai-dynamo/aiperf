<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run a Published Trace (Remote `--input-file`)

`--input-file` accepts a Hugging Face, S3, or HTTPS reference as well as a local path.
AIPerf checks access, pins the exact version, downloads the data into a local cache,
and then replays it exactly as it would the same file on disk.

```bash
aiperf profile --model my-model --url localhost:8000 --endpoint-type chat \
    --input-file hf://llm-d/chat-traces
```

Remote references are for local runs. Kubernetes (`aiperf kube`) support for remote
`--input-file` is not part of this release.

To publish a trace that others can run this way, see [Publishing traces](publishing-traces.md).

---

## Reference Forms

| Source | Form | Notes |
|---|---|---|
| Hugging Face | `hf://[datasets/]<org>/<repo>[@<rev>][/<path>]` | `<rev>` is a branch, tag, or commit; URL-encode slashes (`refs%2Fpr%2F1`). Only dataset repos are supported. |
| S3 | `s3://<bucket>/<key>[?versionId=<id>]` | A key ending in `/` is a prefix, and so is a key that is not an object. Needs `uv pip install 'aiperf[aws]'`. |
| HTTP(S) | `https://...` (or `http://...`) | One file. Presigned S3/GCS URLs work. |

A reference to a **file** runs as is. A reference to a **repo, folder, or prefix** needs an
`aiperf.json` manifest at its root that names the data file and its format; see
[Publishing traces](publishing-traces.md). A single file can also have an optional
`<file>.aiperf.json` sidecar next to it, which supplies the format and other defaults
(HTTPS sidecars are only looked up for URLs without a query string).

Other schemes are rejected when the command line is parsed, and the error lists the supported
ones.

```bash
# A single file in a Hugging Face repo, pinned to a commit
aiperf profile ... --input-file hf://datasets/llm-d/chat-traces@<commit>/data/chat.jsonl.gz \
    --custom-dataset-type mooncake_trace

# An S3 prefix that holds an aiperf.json
aiperf profile ... --input-file s3://my-traces/chat/2026-08/

# A presigned URL
aiperf profile ... --input-file "https://my-bucket.s3.amazonaws.com/chat.jsonl.gz?X-Amz-Signature=..." \
    --custom-dataset-type mooncake_trace
```

`--custom-dataset-type` is optional when a manifest or sidecar declares the format. If you
pass both and they disagree, the run fails before anything is downloaded.

---

## What Happens Before the Run

1. **Access check.** AIPerf checks that the source exists, that your credentials work, and
   that the manifest (if any) is valid and compatible with the run: the format must exist in
   this AIPerf, the `requires.aiperf` version specifier and `requires.endpoint_types` must be
   satisfied. It pins the exact version here (Hugging Face commit SHA, S3 object version or
   ETag, HTTP ETag). Problems fail at this step, before anything is downloaded.
2. **First-record check.** For trace formats, AIPerf reads the first record. If it has
   timestamps, the run replays on the recorded schedule (`--no-fixed-schedule` opts out), just
   like a local trace. To decide this it reads only the leading bytes of the file, in steps of
   64 KiB, 1 MiB, and 8 MiB, and downloads the whole file only when the first record is longer
   than that. Columnar files (Parquet, Arrow) are always read from the downloaded copy.
3. **Download** into the cache, verified against the manifest's `sha256` when present. The
   first record is then checked against the chosen format, so a file that does not look like
   its declared format fails with a clear error instead of during the run.

Options that conflict with a fixed schedule (a request rate, a non-sequential
`--dataset-sampling-strategy`, a parameter sweep) behave as they do for a local file: the
trace is not promoted, and the run fails with the conflict only if the trace turns out to have
timestamps.

A remote `dag_jsonl` file cannot be counted before it is downloaded, so it needs an explicit
stop condition (`--conversation-num`, `--request-count`, or `--benchmark-duration`).

The run log prints the exact version used, for example:

```text
Remote dataset hf://llm-d/chat-traces: using hf://datasets/llm-d/chat-traces@3e1f.../data/chat.jsonl.gz (version 3e1f..., 1 file(s), 48211093 bytes, sha256 9c1d4f0a7b2e)
```

Copy the `using ...` reference to reproduce the run. The same information (reference, pinned
version, downloaded files with their digests, and the manifest digest) is recorded under
`metadata.dataset` in `profile_export_aiperf.json`.

---

## Credentials

| Source | How |
|---|---|
| Hugging Face | `HF_TOKEN`, or `hf auth login`. Gated repos also need their terms accepted on the dataset page. |
| S3 | The standard AWS credential chain: environment variables, `~/.aws`, `aws sso login`, or an instance/pod role. `AWS_ENDPOINT_URL_S3` targets MinIO, Ceph, or R2. |
| HTTPS | None; presigned URLs carry their own. |

Presigned URLs are redacted everywhere AIPerf writes them: logs, error messages,
`run_config.json`, the command line recorded in `profile_export_aiperf.json`, and the
provenance above. A presigned URL that has expired fails with an access-denied error that
suggests generating a new one.

---

## Pinning, Caching, and Offline Runs

- Unpinned references check for a newer version on every run and download only when it changed.
  Pin with `@<commit>` (Hugging Face) or `?versionId=` (S3) for comparable results. An HTTPS URL
  is versioned by its `ETag` (or `Last-Modified` and length); a server that sends neither is
  downloaded again on every run, with a warning.
- Hugging Face files use the standard Hugging Face cache (`HF_HOME`). S3 and HTTPS files go to
  `~/.cache/aiperf/remote` (`AIPERF_DATASET_REMOTE_CACHE_DIR`). Nothing is evicted
  automatically; delete the directory to reclaim disk.
- If another AIPerf process is downloading the same version, the run waits for it. If no other
  run is active, delete the lock file named in the error and retry.
- `AIPERF_DATASET_REMOTE_OFFLINE=1` (or `HF_HUB_OFFLINE=1`) makes AIPerf use only what it has
  already downloaded and never touch the network. An unpinned reference uses the newest cached
  version, and a warning says so. A reference AIPerf has not downloaded before fails with a
  not-found error.
- An interrupted download never leaves a partial file behind, so the next run downloads
  again instead of serving a truncated copy. S3 and HTTPS downloads stop promptly on Ctrl-C.
  Ctrl-C during an `hf://` download waits for the file currently downloading to finish, because
  the Hugging Face client is not interrupted mid-file.
- Transient failures (timeouts, HTTP 429 and 5xx) are retried automatically, three attempts in
  total with a growing delay. If the remote changes after AIPerf pinned it (a force-pushed
  repo, a replaced object), the run fails with a "changed after AIPerf pinned its version"
  error; run again to use the new version.

---

## Troubleshooting

Errors read `Cannot use <reference>: <what went wrong>. <how to fix it>.`

| Error kind | Typical cause | Fix |
|---|---|---|
| `auth_missing` | A private repo or object and no credentials, or no usable AWS credentials | Set `HF_TOKEN` / run `hf auth login`, or configure AWS credentials |
| `auth_denied` | Token lacks permission, AWS credentials expired, or a presigned URL expired | Check permissions; refresh credentials or generate a new presigned URL |
| `gated` | The dataset's terms have not been accepted | Accept them on the dataset page, then retry |
| `not_found` | Wrong repo, path, key, or URL; or offline mode with nothing cached | Check the reference; run once with network access |
| `revision_not_found` | The `@<rev>` does not exist | Check the branch, tag, or commit |
| `manifest_invalid` | A repo, folder, or prefix has no `aiperf.json`, or it is malformed | Ask the publisher to fix or add it |
| `incompatible` | The manifest needs a newer AIPerf or another `--endpoint-type`, or conflicts with your options | Upgrade AIPerf, or change the flag the error names |
| `checksum_mismatch` | The downloaded file does not match the manifest's `sha256` | Retry; if it persists, the publisher's file or manifest is wrong |
| `missing_dependency` | `s3://` without the AWS extra | `uv pip install 'aiperf[aws]'` |
| `transient` | Network trouble or a rate-limited server | Retry later |

If the manifest declares `requires.max_context_tokens`, AIPerf only warns when it is larger
than `--max-context-length`. That comparison happens only when `--max-context-length` is set,
and that option applies to formats that implement context filtering, which today is Weka. It
never fails the run.
