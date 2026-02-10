# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace tokenizer wrapper with sensible defaults."""

import contextlib
import io
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.exceptions import NotInitializedError, TokenizerError

if TYPE_CHECKING:
    from transformers import BatchEncoding

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AliasResolutionResult:
    """Result of tokenizer alias resolution."""

    resolved_name: str
    """The resolved name (canonical ID or original if not resolved)."""

    suggestions: list[tuple[str, int]] = field(default_factory=list)
    """List of (model_id, downloads) suggestions if ambiguous."""

    @property
    def is_ambiguous(self) -> bool:
        """Whether the name was ambiguous (has suggestions but no resolution)."""
        return len(self.suggestions) > 0


class AmbiguousTokenizerNameError(ValueError):
    """Raised when a tokenizer name is ambiguous and has multiple possible matches."""

    def __init__(self, name: str, suggestions: list[tuple[str, int]]) -> None:
        self.name = name
        self.suggestions = suggestions
        super().__init__(
            f"'{name}' is ambiguous. Did you mean: {', '.join(s[0] for s in suggestions[:3])}?"
        )


def _is_offline_mode() -> bool:
    """Check if HuggingFace offline mode is enabled via environment variables."""
    return bool(os.environ.get("HF_HUB_OFFLINE", "")) or bool(
        os.environ.get("TRANSFORMERS_OFFLINE", "")
    )


def resolve_alias(name: str) -> AliasResolutionResult:
    """Resolve a tokenizer name alias to its canonical repository ID.

    Queries the HuggingFace Hub to resolve model aliases
    (e.g., "bert-base-uncased" -> "google-bert/bert-base-uncased").
    Uses HF_TOKEN environment variable for authentication.

    Args:
        name: The tokenizer name or alias to resolve.

    Returns:
        AliasResolutionResult with resolved name and any suggestions.
    """
    # Check if this looks like a local path
    path = Path(name)
    is_local_path = (
        path.is_absolute()
        or name.startswith("./")
        or name.startswith("../")
        or path.is_dir()
    )

    if is_local_path or _is_offline_mode():
        return AliasResolutionResult(resolved_name=name)

    # Lazy import HuggingFace Hub
    from huggingface_hub import list_models, model_info
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    try:
        # Try direct lookup first
        model_info(name)
        # model_info() succeeded — name is a valid HF identifier (possibly
        # a redirect like "gpt2" → "openai-community/gpt2"). Keep the
        # original name so transformers handles the redirect internally and
        # caches under the original name (models--gpt2/, not
        # models--openai-community--gpt2/).
        return AliasResolutionResult(resolved_name=name)
    except (RepositoryNotFoundError, HfHubHTTPError):
        # Search for the model
        try:
            models = list(list_models(search=name, limit=50))
            if not models:
                return AliasResolutionResult(resolved_name=name)

            name_lower = name.lower()
            suffix_matches = []

            for model in models:
                model_id_lower = model.id.lower()
                if model_id_lower == name_lower:
                    return AliasResolutionResult(resolved_name=model.id)
                if model_id_lower.endswith(f"/{name_lower}"):
                    suffix_matches.append(model)

            if suffix_matches:
                suffix_matches.sort(
                    key=lambda m: getattr(m, "downloads", 0) or 0, reverse=True
                )
                return AliasResolutionResult(resolved_name=suffix_matches[0].id)

            # Ambiguous - return suggestions
            sorted_models = sorted(
                models, key=lambda m: getattr(m, "downloads", 0) or 0, reverse=True
            )
            suggestions = [
                (m.id, getattr(m, "downloads", 0) or 0) for m in sorted_models[:5]
            ]
            return AliasResolutionResult(resolved_name=name, suggestions=suggestions)
        except Exception as e:
            _logger.debug(f"Alias search failed for '{name}': {e!r}")
            return AliasResolutionResult(resolved_name=name)
    except Exception as e:
        _logger.debug(f"Alias resolution failed for '{name}': {e!r}")
        return AliasResolutionResult(resolved_name=name)


class Tokenizer:
    """Simplified interface for HuggingFace tokenizers with sensible defaults."""

    def __init__(self) -> None:
        """Initialize with default arguments for call, encode, and decode."""
        self._tokenizer = None
        self._resolved_name: str | None = None
        self._call_args = {"add_special_tokens": False}
        self._encode_args = {"add_special_tokens": False}
        self._decode_args = {"skip_special_tokens": True}

    def _require_init(self) -> None:
        """Raise NotInitializedError if tokenizer is not initialized."""
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")

    @staticmethod
    def resolve_alias(name: str) -> AliasResolutionResult:
        """Resolve a tokenizer name alias to its canonical repository ID."""
        return resolve_alias(name)

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        trust_remote_code: bool = False,
        revision: str = "main",
        resolve_alias: bool = True,
    ) -> "Tokenizer":
        """Load a tokenizer for the given pretrained model name.

        Uses HF_TOKEN environment variable for authentication.

        Args:
            name: The name or path of the pretrained tokenizer model.
            trust_remote_code: Whether to trust remote code when loading.
            revision: The specific model version to use.
            resolve_alias: Whether to resolve model aliases to canonical names.

        Raises:
            AmbiguousTokenizerNameError: If the name is ambiguous.
            TokenizerError: If the tokenizer cannot be loaded.
        """
        try:
            # Silence tokenizer warning on import and first use
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                from transformers import AutoTokenizer

                # Offline mode: skip alias resolution, load from local cache
                if _is_offline_mode():
                    tokenizer_instance = cls._from_pretrained_local(
                        AutoTokenizer.from_pretrained,
                        name,
                        trust_remote_code=trust_remote_code,
                        revision=revision,
                    )
                    tokenizer_instance._resolved_name = name
                    return tokenizer_instance

                # Online mode: resolve alias then load
                resolved_name = name
                if resolve_alias:
                    result = cls.resolve_alias(name)
                    resolved_name = result.resolved_name
                    if result.is_ambiguous:
                        raise AmbiguousTokenizerNameError(name, result.suggestions)

                tokenizer_cls = cls()
                tokenizer_cls._tokenizer = AutoTokenizer.from_pretrained(
                    resolved_name,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                )
                tokenizer_cls._resolved_name = resolved_name
        except AmbiguousTokenizerNameError:
            raise
        except Exception as e:
            raise TokenizerError(
                f"Failed to load tokenizer '{name}'", tokenizer_name=name
            ) from e
        return tokenizer_cls

    @staticmethod
    def _find_cached_model_for_alias(name: str) -> str | None:
        """Scan HF cache for a model whose repo ID ends with /<name>.

        Handles the case where "gpt2" was cached as "openai-community/gpt2"
        (i.e. models--openai-community--gpt2/).

        Returns:
            The full model ID (e.g. "openai-community/gpt2") or None.
        """
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
        if not cache_dir.is_dir():
            return None

        suffix = f"--{name.lower()}"
        for entry in cache_dir.iterdir():
            if (
                entry.is_dir()
                and entry.name.startswith("models--")
                and entry.name.lower().endswith(suffix)
            ):
                # Convert "models--openai-community--gpt2" -> "openai-community/gpt2"
                model_id = entry.name[len("models--") :].replace("--", "/")
                _logger.debug(f"Found cached model for alias '{name}': {model_id}")
                return model_id
        return None

    @classmethod
    def _from_pretrained_local(
        cls,
        from_pretrained_func: Callable,
        name: str,
        trust_remote_code: bool = False,
        revision: str = "main",
    ) -> "Tokenizer":
        """Load a tokenizer from local cache (offline mode)."""
        # Workaround for transformers 4.57+ bug: _patch_mistral_regex
        # calls model_info() even with local_files_only=True
        import huggingface_hub

        class _OfflineModelInfo:
            tags = None

        _original_model_info = huggingface_hub.model_info
        huggingface_hub.model_info = lambda *a, **kw: _OfflineModelInfo()
        try:
            tokenizer_cls = cls()
            try:
                tokenizer_cls._tokenizer = from_pretrained_func(
                    name,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    local_files_only=True,
                )
            except Exception:
                # Cache may be under a resolved alias name (e.g. "gpt2" cached
                # as "openai-community/gpt2"). Scan cache for a match.
                cached_id = cls._find_cached_model_for_alias(name)
                if cached_id is None:
                    raise
                _logger.debug(f"Retrying offline load with cached alias: {cached_id}")
                tokenizer_cls._tokenizer = from_pretrained_func(
                    cached_id,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    local_files_only=True,
                )
            return tokenizer_cls
        finally:
            huggingface_hub.model_info = _original_model_info

    def __call__(self, text, **kwargs) -> "BatchEncoding":
        """
        Call the underlying Huggingface tokenizer with default arguments,
        which can be overridden by kwargs.

        Args:
            text: The input text to tokenize.

        Returns:
            A BatchEncoding object containing the tokenized output.
        """
        self._require_init()
        return self._tokenizer(text, **{**self._call_args, **kwargs})

    def encode(self, text, **kwargs) -> list[int]:
        """
        Encode the input text into a list of token IDs.

        This method calls the underlying Huggingface tokenizer's encode
        method with default arguments, which can be overridden by kwargs.

        Args:
            text: The input text to encode.

        Returns:
            A list of token IDs.
        """
        self._require_init()
        return self._tokenizer.encode(text, **{**self._encode_args, **kwargs})

    def decode(self, token_ids, **kwargs) -> str:
        """
        Decode a list of token IDs back into a string.

        This method calls the underlying Huggingface tokenizer's decode
        method with default arguments, which can be overridden by kwargs.

        Args:
            token_ids: A list of token IDs to decode.

        Returns:
            The decoded string.
        """
        self._require_init()
        return self._tokenizer.decode(token_ids, **{**self._decode_args, **kwargs})

    @property
    def bos_token_id(self) -> int:
        """
        Return the beginning-of-sequence (BOS) token ID.
        """
        self._require_init()
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """
        Return the end-of-sequence (EOS) token ID.
        """
        self._require_init()
        return self._tokenizer.eos_token_id

    @property
    def block_separation_token_id(self) -> int | None:
        """
        Returns BOS, EOS, or None if none are available.
        """
        self._require_init()

        if self.bos_token_id is not None:
            return self.bos_token_id
        if self.eos_token_id is not None:
            return self.eos_token_id
        return None

    def __repr__(self) -> str:
        """
        Return a string representation of the underlying tokenizer.

        Returns:
            The string representation of the tokenizer.
        """
        return self._tokenizer.__repr__()

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the underlying tokenizer.

        Returns:
            The string representation of the tokenizer.
        """
        return self._tokenizer.__str__()
