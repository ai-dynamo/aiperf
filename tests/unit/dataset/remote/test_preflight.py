# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gzip
import logging
from pathlib import Path

import orjson
import pytest

from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.remote import preflight, service, sniff
from aiperf.dataset.remote.cache import entry_dir
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError
from aiperf.dataset.remote.manifest import parse_manifest
from aiperf.dataset.remote.models import ProbeResult, RemoteFile, ResolvedSource
from aiperf.plugin.enums import CustomDatasetType, PhaseType
from tests.unit.conftest import make_run_from_cli

URL = "https://example.test/trace.jsonl"
PRESIGNED = (
    "https://example.test/trace.jsonl?X-Amz-Signature=sekrit123&X-Amz-Credential=cred"
)
TS_ROWS = b'{"timestamp": 0, "input_length": 4, "output_length": 2}\n{"timestamp": 5, "input_length": 4, "output_length": 2}\n'
NO_TS_ROWS = b'{"input_length": 4, "output_length": 2}\n'


class FakeResolver:
    """Serves bytes as a remote https source; every probe sees a new version."""

    def __init__(
        self,
        data: bytes,
        *,
        manifest: dict | None = None,
        filename: str = "trace.jsonl",
    ) -> None:
        self.data, self.manifest, self.filename = data, manifest, filename
        self.probes = 0
        self.prefix_sizes: list[int] = []

    async def probe(self, ref):
        self.probes += 1
        pinned = ref.with_version(f"etag-{self.probes}")
        manifest = (
            parse_manifest(
                orjson.dumps(self.manifest), ref_display=ref.display, kind="sidecar"
            )
            if self.manifest
            else None
        )
        return ProbeResult(ref=pinned, pinned_ref=pinned.display, data_relpath=None, data_is_dir=False,
                           filename=self.filename, manifest=manifest, manifest_sha256=None, needs_auth=False)  # fmt: skip

    async def read_prefix(self, probe, nbytes):
        self.prefix_sizes.append(nbytes)
        return self.data[:nbytes], len(self.data) <= nbytes

    async def materialize(self, probe):
        dest = entry_dir(probe.ref) / probe.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.data)
        return ResolvedSource(probe=probe, local_path=dest, files=[RemoteFile(path=probe.filename, sha256="0" * 64, size_bytes=len(self.data))])  # fmt: skip


@pytest.fixture(autouse=True)
def _reset() -> None:
    preflight.reset_probe_cache()


def _install(monkeypatch, resolver: FakeResolver) -> FakeResolver:
    monkeypatch.setattr(service, "get_resolver", lambda scheme: resolver)
    return resolver


def _no_resolver(scheme: str) -> object:
    raise AssertionError(f"resolver for {scheme} must not be constructed")


def _cfg(input_file: str = URL, **kwargs: object):
    return make_run_from_cli(
        CLIConfig(model_names=["m"], input_file=input_file, **kwargs)
    ).cfg


def _profiling_type(cfg) -> str:
    return str(next(p for p in cfg.phases if p.name == "profiling").type)


def test_timestamped_trace_stays_promoted_and_ref_is_pinned(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS))
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([cfg])
    ds = cfg.get_default_dataset()
    assert ds.path.resolved_version == "etag-1"
    assert ds.remote_fixed_schedule_check is None
    assert "remote_fixed_schedule_check" not in ds.model_fields_set
    assert _profiling_type(cfg) == str(PhaseType.FIXED_SCHEDULE)


def test_untimestamped_trace_is_demoted(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(NO_TS_ROWS))
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([cfg])
    assert _profiling_type(cfg) == str(PhaseType.CONCURRENCY)
    assert cfg.get_default_dataset().remote_fixed_schedule_check is None


def test_manifest_format_drives_promotion_and_block_size(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS, manifest={"aiperf_manifest_version": 1, "format": "mooncake_trace", "block_size": 16}))  # fmt: skip
    cfg = _cfg()
    preflight.preflight_remote_access([cfg])
    ds = cfg.get_default_dataset()
    assert str(ds.format) == "mooncake_trace" and "format" in ds.model_fields_set
    assert ds.block_size == 16
    assert _profiling_type(cfg) == str(PhaseType.FIXED_SCHEDULE)


def test_cli_block_size_wins_over_manifest(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS, manifest={"aiperf_manifest_version": 1, "format": "mooncake_trace", "block_size": 16}))  # fmt: skip
    cfg = _cfg(prompt_input_tokens_block_size=64)
    preflight.preflight_remote_access([cfg])
    assert cfg.get_default_dataset().block_size == 64


def test_untyped_without_manifest_is_demoted_like_local(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS))
    cfg = _cfg()
    preflight.preflight_remote_access([cfg])
    assert _profiling_type(cfg) == str(PhaseType.CONCURRENCY)
    assert "format" not in cfg.get_default_dataset().model_fields_set


def test_manifest_format_conflicting_with_cli_type_is_incompatible(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS, manifest={"aiperf_manifest_version": 1, "format": "bailian_trace"}))  # fmt: skip
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    with pytest.raises(RemoteSourceError, match="bailian_trace") as exc:
        preflight.preflight_remote_access([cfg])
    assert exc.value.kind == RemoteErrorKind.INCOMPATIBLE


def test_manifest_format_is_compared_case_insensitively(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS, manifest={"aiperf_manifest_version": 1, "format": "MOONCAKE_TRACE"}))  # fmt: skip
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([cfg])
    assert cfg.get_default_dataset().format == CustomDatasetType.MOONCAKE_TRACE
    assert _profiling_type(cfg) == str(PhaseType.FIXED_SCHEDULE)


def test_manifest_endpoint_requirement_is_checked(monkeypatch) -> None:
    manifest = {
        "aiperf_manifest_version": 1,
        "requires": {"endpoint_types": ["embeddings"]},
    }
    _install(monkeypatch, FakeResolver(TS_ROWS, manifest=manifest))
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    with pytest.raises(
        RemoteSourceError, match="--endpoint-type embeddings only"
    ) as exc:
        preflight.preflight_remote_access([cfg])
    assert exc.value.kind == RemoteErrorKind.INCOMPATIBLE


def test_manifest_format_rejected_by_dataset_options_is_incompatible(
    monkeypatch,
) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS, manifest={"aiperf_manifest_version": 1, "format": "mooncake_trace"}))  # fmt: skip
    cfg = _cfg(max_context_length=100)
    with pytest.raises(RemoteSourceError, match="mooncake_trace") as exc:
        preflight.preflight_remote_access([cfg])
    assert exc.value.kind == RemoteErrorKind.INCOMPATIBLE


def test_presigned_signature_stays_out_of_manifest_logs(monkeypatch, caplog) -> None:
    manifest = {
        "aiperf_manifest_version": 1,
        "format": "weka_trace",
        "requires": {"max_context_tokens": 1000},
    }
    _install(monkeypatch, FakeResolver(NO_TS_ROWS, manifest=manifest))
    cfg = _cfg(PRESIGNED, max_context_length=100)
    with caplog.at_level(logging.WARNING):
        preflight.preflight_remote_access([cfg])
    warnings = [
        r.getMessage()
        for r in caplog.records
        if "--max-context-length 100" in r.getMessage()
    ]
    assert len(warnings) == 1 and "<redacted>" in warnings[0]
    assert "sekrit123" not in caplog.text


def test_sweep_variations_share_one_probe_and_version(monkeypatch) -> None:
    resolver = _install(monkeypatch, FakeResolver(TS_ROWS))
    cfgs = [
        _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE) for _ in range(3)
    ]
    preflight.preflight_remote_access(cfgs)
    preflight.preflight_remote_materialize(cfgs)
    # FakeResolver hands out a new version per probe, so a second probe anywhere
    # (access or materialize) would pin a different version.
    assert resolver.probes == 1
    assert {c.get_default_dataset().path.resolved_version for c in cfgs} == {"etag-1"}
    assert all(_profiling_type(c) == str(PhaseType.FIXED_SCHEDULE) for c in cfgs)
    assert (
        service.local_path_for(cfgs[2].get_default_dataset().path).read_bytes()
        == TS_ROWS
    )


@pytest.mark.parametrize(
    ("data", "expected"),
    [(TS_ROWS, PhaseType.FIXED_SCHEDULE), (NO_TS_ROWS, PhaseType.CONCURRENCY)],
)  # fmt: skip
def test_every_variant_run_is_finalized(
    monkeypatch, data: bytes, expected: PhaseType
) -> None:
    resolver = _install(monkeypatch, FakeResolver(data))
    cli = CLIConfig(model_names=["m"], input_file=URL, custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    sweep_variants=["a: concurrency=2", "b: concurrency=4"])  # fmt: skip
    plan = build_benchmark_plan(resolve_config(cli, None))
    assert len(plan.configs) == 2
    assert all(
        c.get_default_dataset().remote_fixed_schedule_check for c in plan.configs
    )
    preflight.preflight_remote_access(plan.configs)
    assert resolver.probes == 1
    for cfg in plan.configs:
        ds = cfg.get_default_dataset()
        assert ds.remote_fixed_schedule_check is None
        assert ds.path.resolved_version == "etag-1"
        assert _profiling_type(cfg) == str(expected)


def test_timestamped_trace_with_rate_flag_raises_the_recorded_conflict(
    monkeypatch,
) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS))
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE, request_rate=5)
    with pytest.raises(ConfigurationError, match="incompatible with fixed_schedule"):
        preflight.preflight_remote_access([cfg])


def test_gzip_first_record_is_read_from_a_prefix(monkeypatch) -> None:
    resolver = _install(monkeypatch, FakeResolver(gzip.compress(TS_ROWS * 50_000), filename="trace.jsonl.gz"))  # fmt: skip
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([cfg])
    assert resolver.prefix_sizes == [sniff.PREFIX_STEPS[0]]
    assert _profiling_type(cfg) == str(PhaseType.FIXED_SCHEDULE)


def test_long_first_record_falls_back_to_full_download(monkeypatch) -> None:
    long_first = b'{"pad": "' + b"x" * 5000 + b'", "timestamp": 1}\n'
    resolver = _install(monkeypatch, FakeResolver(long_first + TS_ROWS))
    monkeypatch.setattr(sniff, "PREFIX_STEPS", (64, 256))
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([cfg])
    # Neither prefix holds the whole first line: no "no timestamp" verdict from
    # either, then the full download decides.
    assert resolver.prefix_sizes == [64, 256]
    assert service.cached_source(cfg.get_default_dataset().path).files[
        0
    ].size_bytes == len(resolver.data)
    assert _profiling_type(cfg) == str(PhaseType.FIXED_SCHEDULE)


def test_offline_access_uses_the_cache_without_any_resolver(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS))
    warm = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([warm])
    preflight.preflight_remote_materialize([warm])
    preflight.reset_probe_cache()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(service, "get_resolver", _no_resolver)
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([cfg])
    preflight.preflight_remote_materialize([cfg])
    assert cfg.get_default_dataset().path.resolved_version == "etag-1"
    assert _profiling_type(cfg) == str(PhaseType.FIXED_SCHEDULE)


def test_materialize_downloads_and_logs_the_pinned_version(monkeypatch, caplog) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS))
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([cfg])
    with caplog.at_level(logging.INFO):
        preflight.preflight_remote_materialize([cfg])
    assert (
        service.local_path_for(cfg.get_default_dataset().path).read_bytes() == TS_ROWS
    )
    records = [r for r in caplog.records if r.name == preflight.__name__]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert URL in message and "etag-1" in message and f"{len(TS_ROWS)} bytes" in message


def test_materialize_rejects_a_version_changed_since_access(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(TS_ROWS))
    cfg = _cfg(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    preflight.preflight_remote_access([cfg])
    # A fresh process (no probe cache) re-probes and sees the remote moved on.
    preflight.reset_probe_cache()
    with pytest.raises(RemoteSourceError, match="changed after AIPerf pinned"):
        preflight.preflight_remote_materialize([cfg])


def test_materialize_rejects_first_record_of_the_wrong_format(monkeypatch) -> None:
    _install(monkeypatch, FakeResolver(b'{"definitely": "not mooncake"}\n', manifest={"aiperf_manifest_version": 1, "format": "mooncake_trace"}))  # fmt: skip
    cfg = _cfg(disable_auto_fixed_schedule=True)
    preflight.preflight_remote_access([cfg])
    with pytest.raises(RemoteSourceError, match="mooncake_trace") as exc:
        preflight.preflight_remote_materialize([cfg])
    assert exc.value.kind == RemoteErrorKind.INCOMPATIBLE


def test_local_datasets_are_untouched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(service, "get_resolver", _no_resolver)
    f = tmp_path / "t.jsonl"
    f.write_bytes(TS_ROWS)
    cfg = make_run_from_cli(CLIConfig(model_names=["m"], input_file=str(f))).cfg
    before = cfg.model_dump()
    preflight.preflight_remote_access([cfg])
    preflight.preflight_remote_materialize([cfg])
    assert cfg.model_dump() == before
