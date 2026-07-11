<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Accuracy evaluator environment

`accuracy-worker.txt` is the hash-locked Python 3.12 / x86_64 manylinux 2.28
environment for the long-lived canonical accuracy worker. It pins every
transitive dependency resolved from the project plus the `accuracy` extra.
LiveCodeBench's canonical release configs still use a Hugging Face repository
loading script, while current Lighteval requires a `datasets` release that no
longer executes such scripts. The worker therefore maps the pinned script's
release config to the same immutable raw JSONL files and loads them through
datasets' built-in JSON reader.

The pinned Lighteval 0.13.0 MMLU-Pro endpoint task is run at its authored
zero-shot setting. That task sets each `Doc.instruction` equal to its complete
query, so Lighteval's generic `PromptManager` strips the query from nonzero-shot
examples. The worker rejects `--accuracy-n-shots` values other than zero instead
of producing a silently corrupted prompt.

Create an evaluator environment without changing the Rust runtime environment:

```bash
uv venv --python 3.12 .venv-accuracy
VIRTUAL_ENV=.venv-accuracy uv pip sync requirements/accuracy-worker.txt
VIRTUAL_ENV=.venv-accuracy uv pip install --no-deps -e .

AIPERF_ACCURACY_PYTHON=.venv-accuracy/bin/python \
  cargo run --release -p aiperf -- [BASE_URL] [MODEL] \
  --accuracy-benchmark mmlu-pro --accuracy-max-problems 100
```

Regenerate intentionally with:

```bash
uv pip compile pyproject.toml --extra accuracy \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes -o requirements/accuracy-worker.txt
```

The worker verifies its evaluator-critical direct versions before loading a
benchmark. Its handshake reports all relevant package versions plus the lock
SHA-256. A distribution using another platform should ship an immutable
container and set `AIPERF_ACCURACY_WORKER_IMAGE_DIGEST` instead of silently
resolving a different environment.
