# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp HTTP tokenizer adapter."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from aiperf.common.exceptions import TokenizerError

_KNOWN_ENDPOINT_SUFFIXES = (
    "/v1/chat/completions",
    "/v1/completions",
    "/chat/completions",
    "/completion",
    "/tokenize",
    "/detokenize",
)


def is_http_tokenizer_url(value: str) -> bool:
    """Return whether ``value`` is an absolute HTTP(S) URL."""
    parsed = urlsplit(value)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def normalize_llamacpp_base_url(value: str) -> str:
    """Normalize an inference or tokenizer URL to a llama.cpp API base URL."""
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"llamacpp tokenizer URL must be an absolute http(s) URL, got {value!r}"
        )
    if parsed.username or parsed.password:
        raise ValueError(
            "llamacpp tokenizer URL must not contain credentials; configure "
            "authentication separately"
        )

    path = parsed.path.rstrip("/")
    for suffix in _KNOWN_ENDPOINT_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _parse_token_ids(payload: Any, *, url: str) -> list[int]:
    if not isinstance(payload, dict):
        raise TokenizerError(
            f"llama.cpp tokenizer at '{url}' returned a non-object response"
        )
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in tokens
    ):
        raise TokenizerError(
            f"llama.cpp tokenizer at '{url}' returned an invalid 'tokens' field"
        )
    return tokens


class LlamaCppTokenizerAdapter:
    """Adapt llama.cpp's ``/tokenize`` and ``/detokenize`` routes."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        validate: bool = True,
    ) -> None:
        self.base_url = normalize_llamacpp_base_url(base_url)
        self.name_or_path = self.base_url
        self.bos_token_id: int | None = None
        self.eos_token_id: int | None = None
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        if validate:
            self._validate_api()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TokenizerError(
                f"llama.cpp tokenizer request to '{url}' failed: "
                f"{type(exc).__name__}: {exc}",
                tokenizer_name=self.base_url,
            ) from exc

    def _validate_api(self) -> None:
        probe = "AIPerf tokenizer probe"
        token_ids = self.encode(probe)
        decoded = self.decode(token_ids)
        if decoded != probe:
            raise TokenizerError(
                f"llama.cpp tokenizer at '{self.base_url}' failed the "
                "tokenize/detokenize round-trip probe",
                tokenizer_name=self.base_url,
            )

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Tokenize text through llama.cpp."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        add_special = bool(kwargs.get("add_special_tokens", False))
        payload = self._post(
            "/tokenize",
            {
                "content": text,
                "add_special": add_special,
                "with_pieces": False,
            },
        )
        return _parse_token_ids(payload, url=f"{self.base_url}/tokenize")

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        """Detokenize token IDs through llama.cpp."""
        payload = self._post("/detokenize", {"tokens": token_ids})
        if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
            raise TokenizerError(
                f"llama.cpp tokenizer at '{self.base_url}/detokenize' returned "
                "an invalid 'content' field",
                tokenizer_name=self.base_url,
            )
        return payload["content"]

    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]:
        """Return the Hugging Face-compatible ``input_ids`` shape."""
        if isinstance(text, list):
            return {"input_ids": [self.encode(item, **kwargs) for item in text]}
        return {"input_ids": self.encode(text, **kwargs)}

    def batch_decode(self, token_id_lists: list[list[int]], **kwargs: Any) -> list[str]:
        """Decode multiple token sequences."""
        return [self.decode(token_ids, **kwargs) for token_ids in token_id_lists]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Report zero because AIPerf requests ``add_special=false`` by default."""
        return 0

    @property
    def vocab_size(self) -> int:
        raise TokenizerError(
            "llama.cpp /tokenize and /detokenize do not expose vocab_size; "
            "--prompt-corpus random is not supported with --tokenizer-type llamacpp"
        )

    @property
    def all_special_ids(self) -> list[int]:
        raise TokenizerError(
            "llama.cpp /tokenize and /detokenize do not expose all special token IDs; "
            "--prompt-corpus random is not supported with --tokenizer-type llamacpp"
        )

    def close(self) -> None:
        """Close the owned HTTP connection pool."""
        if self._owns_client:
            self._client.close()
