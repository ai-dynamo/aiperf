# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace tokenizer wrapper with sensible defaults and built-in tiktoken backend."""

import contextlib
import inspect
import io
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.exceptions import NotInitializedError, TokenizerError

if TYPE_CHECKING:
    import tiktoken
    from transformers import BatchEncoding

_logger = logging.getLogger(__name__)

BUILTIN_TOKENIZER_NAME = "builtin"
_BUILTIN_ENCODING = "o200k_base"
# tiktoken encoding names that should be routed through tiktoken, not HF.
# "gpt2" is excluded because it's also a valid HF model name.
TIKTOKEN_ENCODING_NAMES = frozenset(
    {
        "cl100k_base",
        "o200k_base",
        "o200k_harmony",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
    }
)


class _TiktokenAdapter:
    """Adapts tiktoken.Encoding to the interface expected by Tokenizer._tokenizer."""

    def __init__(self, encoding: "tiktoken.Encoding") -> None:
        self._encoding = encoding

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int:
        return self._encoding.eot_token

    def encode(self, text: str, **kwargs) -> list[int]:
        return self._encoding.encode(text, allowed_special="all")

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return self._encoding.decode(token_ids)

    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": self.encode(text)}

    def __repr__(self) -> str:
        return f"TiktokenAdapter({self._encoding.name})"

    def __str__(self) -> str:
        return repr(self)


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


def _supports_kwarg(obj: object, method_name: str, kwarg: str) -> bool:
    """Check if a method on an object accepts a specific keyword argument."""
    method = getattr(obj, method_name, None)
    if method is None:
        return False
    try:
        return kwarg in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _is_offline_mode() -> bool:
    """Check if HuggingFace offline mode is enabled via environment variables."""
    return bool(os.environ.get("HF_HUB_OFFLINE", "")) or bool(
        os.environ.get("TRANSFORMERS_OFFLINE", "")
    )


def _find_hf_cache_aliases(name: str) -> list[Path]:
    """Find HF cache directories matching a model name alias.

    Scans the HF hub cache for ``models--*--<name>`` directories
    (case-insensitive suffix match).

    Returns:
        List of matching cache directory paths.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(HF_HUB_CACHE)
    if not cache_dir.is_dir():
        return []

    suffix = f"--{name.lower()}"
    return [
        entry
        for entry in cache_dir.iterdir()
        if entry.is_dir()
        and entry.name.startswith("models--")
        and entry.name.lower().endswith(suffix)
    ]


def _is_revision_snapshot_cached(model_dir: Path, revision: str) -> bool:
    """Check if a specific revision snapshot exists in an HF model cache directory.

    Supports both named refs (``main``, ``v1.2``) and direct commit hashes.
    """
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    # Named ref: refs/<revision> contains the commit hash
    refs_file = model_dir / "refs" / revision
    if refs_file.is_file():
        commit_hash = refs_file.read_text().strip()
        return (snapshots_dir / commit_hash).is_dir()
    # Direct commit hash
    return (snapshots_dir / revision).is_dir()


def _is_hf_cached(name: str, revision: str | None = None) -> bool:
    """Check if a HuggingFace model is available in the local cache.

    Looks for ``models--<name>/`` (with ``/`` replaced by ``--``) inside the
    HF hub cache directory.  Also handles alias-style short names, returning
    True only when a single unambiguous match exists.

    When *revision* is given, also verifies that the specific revision snapshot
    is present — a model directory from a different revision is not sufficient.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(HF_HUB_CACHE)
    if not cache_dir.is_dir():
        return False

    # Exact match: "meta-llama/Llama-2-7b-hf" -> "models--meta-llama--Llama-2-7b-hf"
    exact = cache_dir / f"models--{name.replace('/', '--')}"
    if exact.is_dir():
        model_dir = exact
    else:
        aliases = _find_hf_cache_aliases(name)
        if len(aliases) != 1:
            return False
        model_dir = aliases[0]

    if revision is None:
        return True
    return _is_revision_snapshot_cached(model_dir, revision)


def _get_revision_snapshot_dir(name: str, revision: str) -> Path | None:
    """Return the cached HF snapshot dir for ``(name, revision)``, or None.

    Resolves alias-style short names the same way ``_is_hf_cached`` does.
    Used by ``_ensure_offline_config_stub`` to locate where a tokenizer-only
    repo's stub ``config.json`` should land.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(HF_HUB_CACHE)
    if not cache_dir.is_dir():
        return None

    exact = cache_dir / f"models--{name.replace('/', '--')}"
    if exact.is_dir():
        model_dir = exact
    else:
        aliases = _find_hf_cache_aliases(name)
        if len(aliases) != 1:
            return None
        model_dir = aliases[0]

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    refs_file = model_dir / "refs" / revision
    if refs_file.is_file():
        snap = snapshots_dir / refs_file.read_text().strip()
    else:
        snap = snapshots_dir / revision
    return snap if snap.is_dir() else None


# tiktoken BPE-file URLs per encoding (sha1(url) keys the disk cache).
# Derived encodings (p50k_edit, o200k_harmony) reuse a base encoding's BPE.
_TIKTOKEN_ENCODING_URLS = {
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "o200k_harmony": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "p50k_base": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    "p50k_edit": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    "r50k_base": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
}


def _is_tiktoken_cached(name: str) -> bool:
    """Check if tiktoken's BPE file is on disk without importing tiktoken.

    Mirrors ``tiktoken.load.read_file_cached`` lookup: ``sha1(url)`` under
    ``TIKTOKEN_CACHE_DIR`` / ``DATA_GYM_CACHE_DIR`` / ``<tempdir>/data-gym-cache``.
    Returns False for unknown encodings or when caching is disabled (empty
    ``TIKTOKEN_CACHE_DIR``) so the caller falls through to a real load —
    must not under-spawn (would leave child processes hitting the no-timeout
    CDN call) but may over-spawn (just wastes a subprocess).
    """
    import hashlib
    import tempfile

    encoding = _BUILTIN_ENCODING if name == BUILTIN_TOKENIZER_NAME else name
    url = _TIKTOKEN_ENCODING_URLS.get(encoding)
    if url is None:
        return False

    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.environ.get("DATA_GYM_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    if cache_dir == "":
        # tiktoken treats empty TIKTOKEN_CACHE_DIR as "disable caching"
        return False

    cache_key = hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()
    return Path(cache_dir, cache_key).is_file()


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

    if is_local_path or _is_offline_mode() or _is_hf_cached(name):
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


# Emitted by transformers when ``tokenizer_config.json`` references a class not
# registered in the installed transformers (e.g. ``TokenizersBackend`` is v5-only).
# Same wording in 4.x and 5.x; if HF rewords it, the hint is silently dropped but
# the original error is still raised.
_MISSING_TOKENIZER_CLASS_RE = re.compile(
    r"Tokenizer class (\S+) does not exist or is not currently imported"
)


def _missing_tokenizer_class_hint(error: Exception) -> str | None:
    """Return an upgrade hint when *error* is HF's missing-class error, else None."""
    match = _MISSING_TOKENIZER_CLASS_RE.search(str(error))
    if match is None:
        return None
    missing = match.group(1)
    try:
        import transformers

        version = transformers.__version__
    except (ImportError, AttributeError):
        version = "<unknown>"
    suggestion = (
        " Try: pip install -U 'transformers>=5'."
        if version.split(".", 1)[0] == "4"
        else ""
    )
    return (
        f"Tokenizer class '{missing}' is not registered in your installed "
        f"transformers {version}.{suggestion}"
    )


class Tokenizer:
    """Simplified interface for HuggingFace tokenizers with sensible defaults."""

    def __init__(self) -> None:
        """Initialize with default arguments for call, encode, and decode."""
        self._tokenizer = None
        self._resolved_name: str | None = None
        self._call_args = {"add_special_tokens": False}
        self._encode_args = {"add_special_tokens": False}
        # Prompt generation inserts BOS/EOS tokens as block separators
        # (see PromptGenerator._build_token_sequence). Skipping special tokens
        # during decode would silently strip those separators.
        self._decode_args = {"skip_special_tokens": False}

    def _require_init(self) -> None:
        """Raise NotInitializedError if tokenizer is not initialized."""
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")

    def _apply_kwarg_overrides(self) -> None:
        """Override default args for tokenizers that use non-standard kwargs (e.g. Kimi)."""
        if self._tokenizer is None:
            return
        if _supports_kwarg(self._tokenizer, "encode", "allow_special_tokens"):
            self._encode_args = {"allow_special_tokens": False}
        elif not _supports_kwarg(self._tokenizer, "encode", "add_special_tokens"):
            self._encode_args = {}

        if _supports_kwarg(self._tokenizer, "__call__", "allow_special_tokens"):
            self._call_args = {"allow_special_tokens": False}
        elif not _supports_kwarg(self._tokenizer, "__call__", "add_special_tokens"):
            self._call_args = {}

        if not _supports_kwarg(self._tokenizer, "decode", "skip_special_tokens"):
            self._decode_args = {}

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
        Pass ``"builtin"`` as *name* for a zero-network-access tokenizer
        backed by tiktoken's ``o200k_base`` encoding.

        Args:
            name: The name or path of the pretrained tokenizer model.
            trust_remote_code: Whether to trust remote code when loading.
            revision: The specific model version to use.
            resolve_alias: Whether to resolve model aliases to canonical names.

        Raises:
            AmbiguousTokenizerNameError: If the name is ambiguous.
            TokenizerError: If the tokenizer cannot be loaded.
        """
        if name == BUILTIN_TOKENIZER_NAME or name in TIKTOKEN_ENCODING_NAMES:
            return cls._from_tiktoken(
                _BUILTIN_ENCODING if name == BUILTIN_TOKENIZER_NAME else name
            )

        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                return cls._load_from_hub(
                    name,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    resolve_alias=resolve_alias,
                )
        except AmbiguousTokenizerNameError:
            raise
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            hint = _missing_tokenizer_class_hint(e)
            if hint:
                detail = f"{detail} {hint}"
            raise TokenizerError(
                f"Failed to load tokenizer '{name}': {detail}",
                tokenizer_name=name,
            ) from e

    @classmethod
    def _load_from_hub(
        cls,
        name: str,
        *,
        trust_remote_code: bool,
        revision: str,
        resolve_alias: bool,
    ) -> "Tokenizer":
        from transformers import AutoTokenizer

        if _is_offline_mode():
            tokenizer_instance = cls._from_pretrained_local(
                AutoTokenizer.from_pretrained,
                name,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            tokenizer_instance._resolved_name = name
            return tokenizer_instance

        # Cache warm (online): skip alias resolution since the cached name is
        # already canonical, but keep ``local_files_only`` off so transformers
        # can perform 404 probes for repo-relative files that legitimately
        # don't exist (tokenizer-only repos lack ``config.json`` —
        # ``local_files_only=True`` mistakes the missing file for a missing
        # cache and surfaces a misleading "Cannot connect to HuggingFace Hub"
        # error).
        if _is_hf_cached(name, revision=revision):
            return cls._build_with_kwargs(
                AutoTokenizer.from_pretrained(
                    name,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                ),
                resolved_name=name,
            )

        resolved_name = name
        if resolve_alias:
            result = cls.resolve_alias(name)
            resolved_name = result.resolved_name
            if result.is_ambiguous:
                raise AmbiguousTokenizerNameError(name, result.suggestions)

        return cls._build_with_kwargs(
            AutoTokenizer.from_pretrained(
                resolved_name,
                trust_remote_code=trust_remote_code,
                revision=revision,
            ),
            resolved_name=resolved_name,
        )

    @classmethod
    def _build_with_kwargs(
        cls, hf_tokenizer: Any, *, resolved_name: str
    ) -> "Tokenizer":
        tokenizer_instance = cls()
        tokenizer_instance._tokenizer = hf_tokenizer
        tokenizer_instance._resolved_name = resolved_name
        tokenizer_instance._apply_kwarg_overrides()
        return tokenizer_instance

    @staticmethod
    def _find_cached_model_for_alias(name: str) -> str | None:
        """Scan HF cache for a model whose repo ID ends with /<name>.

        Handles the case where "gpt2" was cached as "openai-community/gpt2"
        (i.e. models--openai-community--gpt2/).

        Returns:
            The full model ID (e.g. "openai-community/gpt2") or None.
        """
        matches = _find_hf_cache_aliases(name)
        if len(matches) != 1:
            return None
        model_id = matches[0].name[len("models--") :].replace("--", "/")
        _logger.debug(f"Found cached model for alias '{name}': {model_id}")
        return model_id

    @classmethod
    def _from_pretrained_local(
        cls,
        from_pretrained_func: Callable,
        name: str,
        trust_remote_code: bool = False,
        revision: str = "main",
    ) -> "Tokenizer":
        """Load a tokenizer from local cache (offline mode).

        Resolves the cached snapshot directory via ``snapshot_download(
        local_files_only=True)`` and points ``AutoTokenizer`` at that local
        path. Loading via a path skips the ``config.json`` round-trip that
        normally fails for tokenizer-only repos when transformers is forced
        to ``local_files_only=True``.
        """
        # Workaround for transformers 4.57+ bug: _patch_mistral_regex
        # calls model_info() even with local_files_only=True
        import huggingface_hub

        class _OfflineModelInfo:
            tags = None

        _original_model_info = huggingface_hub.model_info
        huggingface_hub.model_info = lambda *a, **kw: _OfflineModelInfo()
        try:
            local_path = cls._resolve_local_snapshot(name, revision)
            tokenizer_cls = cls()
            tokenizer_cls._tokenizer = from_pretrained_func(
                local_path,
                trust_remote_code=trust_remote_code,
            )
            tokenizer_cls._apply_kwarg_overrides()
            return tokenizer_cls
        finally:
            huggingface_hub.model_info = _original_model_info

    @classmethod
    def _resolve_local_snapshot(cls, name: str, revision: str) -> str:
        """Return the on-disk snapshot directory for *name* @ *revision*.

        Tries the canonical repo ID first, then alias matches (e.g. ``gpt2``
        cached as ``openai-community/gpt2``). Raises ``OSError`` with a clear
        message if no local snapshot exists.
        """
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        candidates = [name]
        if "/" not in name:
            cached_alias = cls._find_cached_model_for_alias(name)
            if cached_alias is not None and cached_alias != name:
                candidates.append(cached_alias)

        last_err: Exception | None = None
        for candidate in candidates:
            try:
                return snapshot_download(
                    candidate,
                    revision=revision,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError as e:
                last_err = e
        raise OSError(
            f"Tokenizer '{name}' (revision '{revision}') is not present in the "
            f"HuggingFace cache. Run online once to populate the cache, or unset "
            f"HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE."
        ) from last_err

    @classmethod
    def _from_tiktoken(cls, encoding_name: str = _BUILTIN_ENCODING) -> "Tokenizer":
        """Load a tokenizer backed by tiktoken (no HuggingFace, no network after first cache)."""
        try:
            import tiktoken
        except ImportError as e:
            raise TokenizerError(
                f"tiktoken is required for --tokenizer {encoding_name}",
                tokenizer_name=encoding_name,
            ) from e

        tokenizer_cls = cls()
        tokenizer_cls._tokenizer = _TiktokenAdapter(
            tiktoken.get_encoding(encoding_name)
        )
        tokenizer_cls._resolved_name = encoding_name
        tokenizer_cls._call_args = {}
        tokenizer_cls._encode_args = {}
        tokenizer_cls._decode_args = {}
        return tokenizer_cls

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
    def resolved_name(self) -> str | None:
        """The resolved model name used to load this tokenizer."""
        return self._resolved_name

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
