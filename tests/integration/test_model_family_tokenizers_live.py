# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf must keep working on current model families (AIP-1150).

The doc end-to-end suite only ever exercises models small enough to serve on a
22 GiB CI device, so in practice everything is benchmarked against Qwen3-0.6B.
Nothing tells us when a new family breaks AIPerf -- a user does.

Most of that breakage is not in serving, it is in the tokenizer: a format
`transformers` has not seen, a repo that requires `trust_remote_code`, a missing
or changed chat template. None of that needs a GPU or the weights. Tokenizer
files are a few MB, so every family on the list below can be checked on an
ordinary runner in seconds -- which is the only reason covering current families
is affordable at all, given the smallest non-Qwen member upstream is ~31B.

These make REAL network calls to HuggingFace. Run with:
    uv run pytest tests/integration/test_model_family_tokenizers_live.py -m integration
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest import param

from aiperf.common.tokenizer import Tokenizer

pytestmark = [pytest.mark.integration, pytest.mark.network]


@dataclass(frozen=True)
class Family:
    """One model family, pinned to a member whose tokenizer we check."""

    family: str
    model: str
    trust_remote_code: bool
    """Whether loading REQUIRES it. Pinned both ways: a family that starts or
    stops needing it is a user-visible change, because AIPerf defaults the flag
    off and a run simply fails until `--tokenizer-trust-remote-code` is passed."""
    has_chat_template: bool
    """Whether the repo ships a chat template. Pinned because the chat endpoint
    has nothing to render without one."""


# Derived from the model families carried in ai-dynamo/dynamo `recipes/`, which
# is curated and updated as families ship -- it grew from 12 to ~31 recipes in
# five months. A hand-maintained list is what left AIPerf on four stale models.
FAMILIES = [
    Family("qwen", "Qwen/Qwen3-0.6B", False, True),
    Family("deepseek", "deepseek-ai/DeepSeek-R1", False, True),
    Family("glm", "zai-org/GLM-5.2-FP8", False, True),
    Family("gemma", "nvidia/Gemma-4-31B-IT-NVFP4", False, True),
    Family("gpt-oss", "openai/gpt-oss-120b", False, True),
    Family("ax", "skt/A.X-K2", False, True),
    Family("solar-open2", "nota-ai/Solar-Open2-250B-Nota-NVFP4", False, True),
    Family("k-exaone", "LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-NVFP4", False, True),
    Family("inkling", "thinkingmachines/Inkling-NVFP4", False, True),
    Family("nemotron", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8", False, True),
    # Kimi is the outlier on both axes: its tokenizer refuses to load without
    # trust_remote_code, and it ships no chat template at all.
    Family("kimi", "moonshotai/Kimi-K3", True, False),
]

CASES = [param(f, id=f.family) for f in FAMILIES]  # fmt: skip


def _load(model: str, *, trust_remote_code: bool) -> Tokenizer:
    try:
        return Tokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception as e:  # noqa: BLE001
        text = str(e).lower()
        if any(k in text for k in ("connection", "timeout", "resolve", "network")):
            pytest.skip(f"HuggingFace unreachable: {e}")
        raise


@pytest.mark.parametrize("fam", CASES)
def test_family_tokenizer_loads(fam: Family) -> None:
    """The baseline: AIPerf can load this family's tokenizer at all."""
    tokenizer = _load(fam.model, trust_remote_code=fam.trust_remote_code)
    assert tokenizer.encode("hello world"), "tokenizer produced no tokens"


@pytest.mark.parametrize("fam", CASES)
def test_family_round_trips_text(fam: Family) -> None:
    tokenizer = _load(fam.model, trust_remote_code=fam.trust_remote_code)
    text = "The quick brown fox jumps over the lazy dog."
    assert text in tokenizer.decode(tokenizer.encode(text))


@pytest.mark.parametrize("fam", CASES)
def test_trust_remote_code_requirement_is_unchanged(fam: Family) -> None:
    """Pinned in both directions, because the default is off.

    A family that starts requiring `trust_remote_code` breaks every user command
    for it until `--tokenizer-trust-remote-code` is added, with no warning from
    us. One that stops requiring it means we are asking users to enable remote
    code execution they no longer need.
    """
    needed = True
    try:
        Tokenizer.from_pretrained(fam.model, trust_remote_code=False)
        needed = False
    except Exception as e:  # noqa: BLE001
        text = str(e).lower()
        if any(k in text for k in ("connection", "timeout", "resolve", "network")):
            pytest.skip(f"HuggingFace unreachable: {e}")

    assert needed == fam.trust_remote_code, (
        f"{fam.family} ({fam.model}) now "
        f"{'requires' if needed else 'does not require'} trust_remote_code; "
        f"the pinned expectation says {fam.trust_remote_code}. AIPerf defaults "
        "--tokenizer-trust-remote-code off, so this changes what a user must pass."
    )


@pytest.mark.parametrize("fam", CASES)
def test_chat_template_availability_is_unchanged(fam: Family) -> None:
    """The chat endpoint has nothing to render without a template."""
    tokenizer = _load(fam.model, trust_remote_code=fam.trust_remote_code)
    hf = tokenizer._tokenizer
    present = getattr(hf, "chat_template", None) is not None

    assert present == fam.has_chat_template, (
        f"{fam.family} ({fam.model}) chat template "
        f"{'appeared' if present else 'disappeared'}; pinned expectation is "
        f"{fam.has_chat_template}"
    )

    if not present:
        return
    rendered = hf.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered, f"{fam.family} chat template rendered empty output"
