# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection of LLM-hallucinated placeholder model names.

When `--tokenizer` is unset, AIPerf normally derives the tokenizer name from
`--model`. If the model name is an obvious placeholder (`mock-model`,
`test-model`, `fake-llama`, ...), an HF Hub lookup fails or ambiguously
matches an unrelated repo. The early tokenizer validator uses
:func:`is_fake_model_name` to substitute the `builtin` tokenizer instead,
with a single warning, so a user iterating against a mock or test server
gets clean token counts rather than a confusing tokenizer error.
"""

__all__ = ["is_fake_model_name"]

# Exact-match set: bare placeholder words used as the entire model name.
_FAKE_MODEL_EXACT = frozenset(
    {
        "test",
        "mock",
        "fake",
        "dummy",
        "example",
        "sample",
        "placeholder",
    }
)

# Substring set: compound markers safe enough not to false-positive on real
# names that merely contain "test"/"mock". Each marker requires a separator
# (`-`) so e.g. "contestant" does not match.
_FAKE_MODEL_SUBSTRINGS = (
    "mock-",
    "-mock",
    "fake-",
    "-fake",
    "test-model",
    "-test-model",
    "your-model",
    "my-model",
    "model-name",
    "model-id",
)


def is_fake_model_name(name: str) -> bool:
    """Return True if ``name`` looks like an LLM-hallucinated placeholder.

    Detection rule:
      1. Reject path-like input — names containing ``/`` or ``\\``, or
         starting with ``.`` or ``~``, are never placeholders.
      2. Lowercase, and replace ``_`` with ``-``.
      3. Match against an exact-name set or a substring set.

    Example:
        >>> is_fake_model_name("mock-llama")
        True
        >>> is_fake_model_name("Test_Model_v2")
        True
        >>> is_fake_model_name("Qwen/Qwen3-0.6B")
        False
        >>> is_fake_model_name("./local-model")
        False
    """
    if not name:
        return False
    if "/" in name or "\\" in name or name[0] in (".", "~"):
        return False
    normalized = name.lower().replace("_", "-")
    if normalized in _FAKE_MODEL_EXACT:
        return True
    return any(token in normalized for token in _FAKE_MODEL_SUBSTRINGS)
