# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for parallel_decode module."""

import importlib
import os
from unittest.mock import MagicMock, call, patch

import pytest

from aiperf.dataset.generator.parallel_decode import _set_daemon, parallel_decode

# Import the module directly (not through __init__.py which exports the function)
pd_module = importlib.import_module("aiperf.dataset.generator.parallel_decode")


class TestParallelDecode:
    """Test suite for parallel_decode module."""

    def test_parallel_decode_empty_list(self):
        """Test parallel_decode with empty input returns empty list."""
        result = parallel_decode([], "gpt2")
        assert result == []

    @patch("aiperf.common.tokenizer.Tokenizer")
    def test_parallel_decode_small_batch_sequential(self, mock_tokenizer_class):
        """Test that small batches (< 10) use sequential decoding."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "decoded"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        token_sequences = [[1, 2, 3], [4, 5, 6]]  # Less than 10
        result = parallel_decode(token_sequences, "gpt2")

        # Should use sequential decoding (Tokenizer.from_pretrained called once)
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "gpt2",
            trust_remote_code=False,
            revision="main",
            resolve_alias=False,
        )
        assert mock_tokenizer.decode.call_count == 2
        assert result == ["decoded", "decoded"]

    @patch.object(pd_module, "ProcessPoolExecutor")
    def test_parallel_decode_large_batch_uses_executor(self, mock_executor_class):
        """Test that large batches (>= 10) use ProcessPoolExecutor."""
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = ["decoded"] * 15
        mock_executor_class.return_value = mock_executor

        token_sequences = [[i] for i in range(15)]  # 15 sequences
        result = parallel_decode(token_sequences, "gpt2")

        # Should use ProcessPoolExecutor
        mock_executor_class.assert_called_once()
        mock_executor.map.assert_called_once()
        assert len(result) == 15

    @patch.object(pd_module, "mp")
    @patch.object(pd_module, "ProcessPoolExecutor")
    def test_parallel_decode_respects_max_workers(self, mock_executor_class, mock_mp):
        """Test that max_workers parameter is respected."""
        mock_mp.cpu_count.return_value = 16
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = ["decoded"] * 15
        mock_executor_class.return_value = mock_executor

        token_sequences = [[i] for i in range(15)]
        parallel_decode(token_sequences, "gpt2", max_workers=4)

        # Should be called with max_workers=4
        call_kwargs = mock_executor_class.call_args.kwargs
        assert call_kwargs["max_workers"] == 4

    @patch.object(pd_module, "mp")
    @patch.object(pd_module, "ProcessPoolExecutor")
    def test_parallel_decode_default_max_workers_capped_at_8(
        self, mock_executor_class, mock_mp
    ):
        """Test that default max_workers is capped at 8."""
        mock_mp.cpu_count.return_value = 64  # Lots of CPUs
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = ["decoded"] * 15
        mock_executor_class.return_value = mock_executor

        token_sequences = [[i] for i in range(15)]
        parallel_decode(token_sequences, "gpt2")

        # Should be capped at 8
        call_kwargs = mock_executor_class.call_args.kwargs
        assert call_kwargs["max_workers"] == 8


class TestSetDaemon:
    """Test suite for _set_daemon helper."""

    def test_set_daemon_uses_property(self):
        """_set_daemon sets daemon via the public property when possible."""
        mock_proc = MagicMock()
        mock_proc.daemon = True
        with patch.object(pd_module.mp, "current_process", return_value=mock_proc):
            _set_daemon(False)
        assert mock_proc.daemon is False

    def test_set_daemon_falls_back_to_config_on_assertion_error(self):
        """_set_daemon falls back to _config when property raises AssertionError."""
        mock_proc = MagicMock()
        type(mock_proc).daemon = property(
            fget=lambda self: self._config.get("daemon"),
            fset=MagicMock(side_effect=AssertionError),
        )
        mock_proc._config = {"daemon": True}
        with patch.object(pd_module.mp, "current_process", return_value=mock_proc):
            _set_daemon(False)
        assert mock_proc._config["daemon"] is False


class TestParallelDecodeDaemonFlag:
    """Test that parallel_decode properly manages the daemon flag."""

    @patch.object(pd_module, "ProcessPoolExecutor")
    def test_daemon_flag_cleared_before_executor_and_restored_after(
        self, mock_executor_class
    ):
        """Daemon flag is cleared before spawning and restored after."""
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = ["decoded"] * 15
        mock_executor_class.return_value = mock_executor

        mock_process = MagicMock()
        mock_process.daemon = True
        with (
            patch.object(pd_module.mp, "current_process", return_value=mock_process),
            patch.object(pd_module, "_set_daemon") as mock_set,
        ):
            parallel_decode([[i] for i in range(15)], "gpt2")

        assert mock_set.call_args_list == [call(False), call(True)]

    @patch.object(pd_module, "ProcessPoolExecutor")
    def test_daemon_flag_not_toggled_for_non_daemon_process(self, mock_executor_class):
        """Daemon flag is not toggled when the process is not a daemon."""
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = ["decoded"] * 15
        mock_executor_class.return_value = mock_executor

        mock_process = MagicMock()
        mock_process.daemon = False
        with (
            patch.object(pd_module.mp, "current_process", return_value=mock_process),
            patch.object(pd_module, "_set_daemon") as mock_set,
        ):
            parallel_decode([[i] for i in range(15)], "gpt2")

        mock_set.assert_not_called()

    @patch.object(pd_module, "ProcessPoolExecutor")
    def test_daemon_flag_restored_on_executor_error(self, mock_executor_class):
        """Daemon flag is restored even when the executor raises."""
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.side_effect = RuntimeError("boom")
        mock_executor_class.return_value = mock_executor

        mock_process = MagicMock()
        mock_process.daemon = True
        with (
            patch.object(pd_module.mp, "current_process", return_value=mock_process),
            patch.object(pd_module, "_set_daemon") as mock_set,
            pytest.raises(RuntimeError, match="boom"),
        ):
            parallel_decode([[i] for i in range(15)], "gpt2")

        assert mock_set.call_args_list == [call(False), call(True)]


class TestWorkerFunctions:
    """Test suite for worker functions."""

    def test_decode_tokens_raises_without_init(self):
        """Test that _decode_tokens raises if worker not initialized."""
        pd_module._worker_tokenizer = None

        with pytest.raises(RuntimeError, match="not initialized"):
            pd_module._decode_tokens([1, 2, 3])

    @patch("aiperf.common.tokenizer.Tokenizer")
    def test_init_worker_loads_tokenizer(self, mock_tokenizer_class):
        """Test that _init_worker loads the tokenizer."""
        pd_module._worker_tokenizer = None
        pd_module._worker_tokenizer_key = None

        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        pd_module._init_worker("gpt2")

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "gpt2",
            trust_remote_code=False,
            revision="main",
            resolve_alias=False,
        )
        assert pd_module._worker_tokenizer is mock_tokenizer
        assert pd_module._worker_tokenizer_key == ("gpt2", False, "main")

    @patch("aiperf.common.tokenizer.Tokenizer")
    def test_init_worker_sets_offline_mode(self, mock_tokenizer_class, monkeypatch):
        """Test that _init_worker enables HuggingFace offline mode."""
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        pd_module._worker_tokenizer = None
        pd_module._worker_tokenizer_key = None

        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        pd_module._init_worker("gpt2")

        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    @patch("aiperf.common.tokenizer.Tokenizer")
    def test_init_worker_reuses_tokenizer_same_name(self, mock_tokenizer_class):
        """Test that _init_worker reuses tokenizer if same name."""
        mock_tokenizer = MagicMock()
        pd_module._worker_tokenizer = mock_tokenizer
        pd_module._worker_tokenizer_key = ("gpt2", False, "main")

        pd_module._init_worker("gpt2")

        # Should NOT call from_pretrained again
        mock_tokenizer_class.from_pretrained.assert_not_called()
        assert pd_module._worker_tokenizer is mock_tokenizer

    @patch("aiperf.common.tokenizer.Tokenizer")
    def test_init_worker_reloads_tokenizer_different_name(self, mock_tokenizer_class):
        """Test that _init_worker reloads tokenizer if different name."""
        old_tokenizer = MagicMock()
        pd_module._worker_tokenizer = old_tokenizer
        pd_module._worker_tokenizer_key = ("gpt2", False, "main")

        new_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = new_tokenizer

        pd_module._init_worker("llama")

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "llama",
            trust_remote_code=False,
            revision="main",
            resolve_alias=False,
        )
        assert pd_module._worker_tokenizer is new_tokenizer
        assert pd_module._worker_tokenizer_key == ("llama", False, "main")

    def test_decode_tokens_uses_worker_tokenizer(self):
        """Test that _decode_tokens uses the worker tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "decoded text"
        pd_module._worker_tokenizer = mock_tokenizer

        result = pd_module._decode_tokens([1, 2, 3])

        mock_tokenizer.decode.assert_called_once_with([1, 2, 3])
        assert result == "decoded text"

    @patch("aiperf.common.tokenizer.Tokenizer")
    def test_init_worker_passes_trust_remote_code_and_revision(
        self, mock_tokenizer_class
    ):
        """Test that _init_worker forwards trust_remote_code and revision."""
        pd_module._worker_tokenizer = None
        pd_module._worker_tokenizer_key = None
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        pd_module._init_worker("kimi-vl", trust_remote_code=True, revision="v1.2")

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "kimi-vl",
            trust_remote_code=True,
            revision="v1.2",
            resolve_alias=False,
        )


class TestParallelDecodeTokenizerArgs:
    """Test that parallel_decode passes tokenizer args through."""

    @patch("aiperf.common.tokenizer.Tokenizer")
    def test_small_batch_passes_trust_remote_code_and_revision(
        self, mock_tokenizer_class
    ):
        """Test sequential path forwards trust_remote_code and revision."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "decoded"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        parallel_decode(
            [[1, 2]],
            "kimi-vl",
            trust_remote_code=True,
            revision="v1.2",
        )

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "kimi-vl",
            trust_remote_code=True,
            revision="v1.2",
            resolve_alias=False,
        )

    @patch.object(pd_module, "ProcessPoolExecutor")
    def test_large_batch_passes_trust_remote_code_and_revision_to_workers(
        self, mock_executor_class
    ):
        """Test executor path forwards trust_remote_code and revision via initargs."""
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = ["decoded"] * 15
        mock_executor_class.return_value = mock_executor

        parallel_decode(
            [[i] for i in range(15)],
            "kimi-vl",
            trust_remote_code=True,
            revision="v1.2",
        )

        call_kwargs = mock_executor_class.call_args.kwargs
        assert call_kwargs["initargs"] == ("kimi-vl", True, "v1.2")
