# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.environment import Environment
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.config.flags import CLIConfig
from aiperf.dataset import mmap_cache
from aiperf.dataset.composer.custom import CustomDatasetComposer
from aiperf.dataset.remote.cache import entry_dir
from aiperf.dataset.remote.index import write_index
from aiperf.dataset.remote.models import ProbeResult, RemoteFile, ResolvedSource
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli

URL = "https://example.test/trace.jsonl"
ROWS = b'{"timestamp": 0, "input_length": 4, "output_length": 2, "hash_ids": [1]}\n'


@pytest.fixture(autouse=True)
def _mmap_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_DIR", tmp_path / "mmap")
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", True)


def _index(ref: RemoteRef, local: Path, filename: str, size: int) -> None:
    probe = ProbeResult(ref=ref, pinned_ref=URL, data_relpath=None, data_is_dir=False, filename=filename,
                        manifest=None, manifest_sha256=None, needs_auth=False)  # fmt: skip
    write_index(
        ResolvedSource(
            probe=probe,
            local_path=local,
            files=[RemoteFile(path=filename, sha256="0" * 64, size_bytes=size)],
        )
    )


def _materialized(version: str, data: bytes = ROWS) -> tuple[RemoteRef, Path]:
    ref = RemoteRef(ref=URL).with_version(version)
    local = entry_dir(ref) / "trace.jsonl"
    local.parent.mkdir(parents=True)
    local.write_bytes(data)
    _index(ref, local, "trace.jsonl", len(data))
    return ref, local


def _run_with(
    ref: RemoteRef, dataset_type: CustomDatasetType = CustomDatasetType.MOONCAKE_TRACE
):
    run = make_run_from_cli(
        CLIConfig(model_names=["m"], input_file=URL, custom_dataset_type=dataset_type, disable_auto_fixed_schedule=True)
    )  # fmt: skip
    ds = run.cfg.datasets[0]
    run.cfg.datasets[0] = ds.model_copy(update={"path": ref})
    return run


def test_dataset_resolver_uses_cached_path_and_sniffs_timing() -> None:
    ref, local = _materialized("v1")
    run = _run_with(ref)
    DatasetResolver().resolve(run)
    assert run.resolved.dataset_file_paths["main"] == local.absolute()
    assert run.resolved.dataset_has_timing_data["main"] is True


def test_dataset_resolver_keeps_symlink_name_of_cached_file() -> None:
    """HF cache files are symlinks to suffixless blobs; resolving them would lose ``.parquet``."""
    ref = RemoteRef(ref=URL).with_version("v1")
    folder = entry_dir(ref)
    folder.mkdir(parents=True)
    blob = folder / "blobs" / "0123abcd"
    blob.parent.mkdir()
    blob.write_bytes(ROWS)
    link = folder / "trace.parquet"
    try:
        link.symlink_to(blob)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform")
    _index(ref, link, "trace.parquet", len(ROWS))

    run = _run_with(ref)
    DatasetResolver().resolve(run)

    resolved = run.resolved.dataset_file_paths["main"]
    assert resolved == link.absolute()
    assert resolved.suffix == ".parquet"
    assert resolved.is_symlink()


def test_composer_reads_cached_path() -> None:
    ref, local = _materialized("v1")
    composer = CustomDatasetComposer(
        run=_run_with(ref, CustomDatasetType.SINGLE_TURN), tokenizer=MagicMock()
    )
    assert composer._file_path == str(local)


def test_mmap_key_hashes_remote_content() -> None:
    ref_a, _ = _materialized("va")
    a = mmap_cache.compute_cache_key_from_run(_run_with(ref_a))
    a_again = mmap_cache.compute_cache_key_from_run(
        _run_with(RemoteRef(ref=URL).with_version("va"))
    )
    ref_b, _ = _materialized("vb", ROWS * 2)
    b = mmap_cache.compute_cache_key_from_run(_run_with(ref_b))
    assert a is not None
    assert a == a_again != b


def test_unmaterialized_ref_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="never resolved"):
        mmap_cache.compute_cache_key_from_run(_run_with(RemoteRef(ref=URL)))
