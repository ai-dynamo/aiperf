<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run a Published Trace (Remote `--input-file`)

`--input-file` accepts a Hugging Face, S3, or HTTPS reference as well as a local path.
AIPerf checks access, pins the exact version, downloads the data into a local cache,
and then replays it exactly as it would the same file on disk.

```bash
# hf://<org>/<repo> is a placeholder; use a real published trace
aiperf profile --model my-model --url localhost:8000 --endpoint-type chat \
    --input-file hf://<org>/<repo>
```

Remote references are for local runs. Kubernetes runs (`aiperf kube` commands, and
`AIPerfJob`/`AIPerfSweep` resources applied to the operator) reject them before anything is
scheduled, with "Remote dataset references (hf://, s3://, https://) are not yet supported for
Kubernetes runs". Download the file and use a local path instead.

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
aiperf profile ... --input-file hf://datasets/<org>/<repo>@<commit>/data/chat.jsonl.gz \
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
Remote dataset hf://<org>/<repo>: using hf://datasets/<org>/<repo>@3e1f.../data/chat.jsonl.gz (version 3e1f..., 1 file(s), 48211093 bytes, sha256 9c1d4f0a7b2e)
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

Errors read `Cannot use <reference>: <what went wrong>. <how to fix it>.` Find your message
in the table.

| The message says | Typical cause | Fix |
|---|---|---|
| "requires authentication (HTTP 401)", "was not found; it may be private", or "no usable AWS credentials" | A private repo or object and no credentials | Set `HF_TOKEN` / run `hf auth login`, or configure AWS credentials |
| "the AWS credentials are expired or invalid" | Expired AWS session | Refresh them (for example `aws sso login`) |
| "denied access" or "S3 denied access" | The token or IAM role lacks permission, or an HTTPS presigned URL expired | Check permissions, or generate a new presigned URL. Without `s3:ListBucket`, S3 also answers this for a key that does not exist |
| "has not accepted its terms" | A gated Hugging Face dataset | Accept the terms on the dataset page, then retry |
| "has no revision" or "has no version" | The `@<rev>` or `?versionId=` does not exist | Check the branch, tag, commit, or version id |
| "was not found, or your token can't access it", "S3 has no such bucket or key", or "the server has no such file" | Wrong repo, path, key, or URL | Check the reference |
| "offline mode is on and it is not in the local cache" | Offline mode with nothing downloaded for that reference | Run once with network access, or unset `AIPERF_DATASET_REMOTE_OFFLINE` / `HF_HUB_OFFLINE` |
| "no `aiperf.json` describing it" or "no `aiperf.json` under it" | A repo, folder, or prefix without a manifest | Point at a data file, or ask the publisher to add `aiperf.json` |
| "its manifest points at" ... "which does not exist", or "its manifest" ... "is invalid" | A malformed or stale manifest | Ask the publisher to fix `aiperf.json` |
| "under the data prefix cannot be stored as a local file" | An S3 key under the prefix contains `..`, `.`, an empty segment, or a backslash, or (on Windows) a colon or drive letter | Ask the publisher to rename that object |
| "it requires AIPerf", "newer than this AIPerf supports", or "does not have" | The manifest needs a newer AIPerf | Upgrade AIPerf |
| "it supports --endpoint-type" | The manifest limits the endpoint types | Pass a supported `--endpoint-type` |
| "its manifest declares format" or "conflicts with this run's dataset options" | The manifest disagrees with your flags | Drop the conflicting option, or pass the manifest's format |
| "its first record does not look like" | The file is not in the declared format | Check the manifest's format, or pass the right `--custom-dataset-type` |
| "the downloaded file's sha256 is" | The file does not match the manifest's `sha256` | Retry; if it persists, the publisher's file or manifest is wrong |
| "changed after AIPerf pinned its version" | The remote changed during the run | Run again to use the new version |
| "need the optional botocore dependency" | `s3://` without the AWS extra | `uv pip install 'aiperf[aws]'` |
| "are not yet supported for Kubernetes runs" | A remote reference passed to `aiperf kube` or set in an `AIPerfJob`/`AIPerfSweep` | Download the file and use a local path |
| "cannot be written" | The cache directory is full or read-only | Free disk space or fix permissions, or set `AIPERF_DATASET_REMOTE_CACHE_DIR` (Hugging Face: `HF_HOME`) |
| "the connection failed", "is unavailable", or "could not be reached" | Network trouble or a rate-limited server | Check network access and retry later |
| "timed out waiting for another process" | Another AIPerf run holds the download lock | Wait, or delete the lock file the message names if no run is active |

If the manifest declares `requires.max_context_tokens`, AIPerf only warns when it is larger
than `--max-context-length`. That comparison happens only when `--max-context-length` is set,
and that option applies to formats that implement context filtering, which today is Weka. It
never fails the run.
