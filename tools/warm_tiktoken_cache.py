#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-download tiktoken BPE encoding files into ``TIKTOKEN_CACHE_DIR``.

tiktoken fetches encoding files from ``openaipublic.blob.core.windows.net``
on first use and performs no retries; a transient CDN outage (observed as
``503 Server Error: The server is busy``) fails every test that touches a
tokenizer in that window. CI runs this script after dependency install, with
``TIKTOKEN_CACHE_DIR`` pointing at an ``actions/cache``-persisted directory,
so test runs never hit the network: on a cache hit ``tiktoken.get_encoding``
is a pure disk read, and on a cache miss the download happens here, once,
with retries and backoff instead of mid-test.

The four names below cover every URL in
``aiperf.common.tokenizer._TIKTOKEN_ENCODING_URLS``; the derived encodings
(``p50k_edit``, ``o200k_harmony``) reuse a base encoding's BPE file and are
served by the same sha1(url)-keyed cache entries.
"""

import os
import sys
import time

ENCODINGS = ("cl100k_base", "o200k_base", "p50k_base", "r50k_base")
MAX_ATTEMPTS = 6


def main() -> int:
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_dir:
        print(
            "TIKTOKEN_CACHE_DIR is not set; refusing to warm an implicit "
            "tempdir cache (set it to the directory CI persists).",
            file=sys.stderr,
        )
        return 1
    os.makedirs(cache_dir, exist_ok=True)

    import tiktoken

    for name in ENCODINGS:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                tiktoken.get_encoding(name)
                print(f"warmed {name}")
                break
            except Exception as exc:  # noqa: BLE001 - retry any fetch failure
                if attempt == MAX_ATTEMPTS:
                    print(
                        f"failed to warm {name} after {MAX_ATTEMPTS} attempts: {exc!r}",
                        file=sys.stderr,
                    )
                    return 1
                delay = 2**attempt
                print(
                    f"{name}: attempt {attempt} failed ({exc!r}); retrying in {delay}s"
                )
                time.sleep(delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
