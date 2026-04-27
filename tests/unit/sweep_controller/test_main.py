# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson

from aiperf.sweep_controller.main import (
    AGGREGATE_READY_MARKER,
    _load_aggregate_for_cr,
    aggregate_marker_exists,
    write_aggregate_marker,
)


def test_aggregate_marker_lifecycle(tmp_path: Path):
    base = tmp_path / "results"
    base.mkdir()
    assert aggregate_marker_exists(base) is False
    write_aggregate_marker(base)
    assert aggregate_marker_exists(base) is True
    assert (base / AGGREGATE_READY_MARKER).exists()


def test_aggregate_marker_atomic_rename(tmp_path: Path):
    """Marker is written via .tmp + rename; partial writes don't appear ready."""
    base = tmp_path
    write_aggregate_marker(base)
    assert (base / AGGREGATE_READY_MARKER).exists()
    # No leftover .tmp
    assert not (base / (AGGREGATE_READY_MARKER + ".tmp")).exists()


def _write_json(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(doc))


def test_load_aggregate_for_cr_loads_all_three_keys(tmp_path: Path):
    """Small bundle: parent + children + confidence all inlined."""
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "s" / "1234"
    _write_json(sweep_dir / "aggregate.json", {"parent": "ok"})
    _write_json(sweep_dir / "children.json", [{"name": "c1"}])
    _write_json(
        base_dir / "aggregate" / "profile_export_aiperf_aggregate.json", {"k": "v"}
    )

    bundle = _load_aggregate_for_cr(base_dir, "ns", "s", "1234")

    assert bundle["parent"] == {"parent": "ok"}
    assert bundle["children"] == [{"name": "c1"}]
    assert bundle["confidence"] == {"k": "v"}


def test_load_aggregate_for_cr_drops_confidence_when_over_size_cap(
    tmp_path: Path, monkeypatch
):
    """Bundle over the inline cap drops `confidence` to keep CR patch < 1MB.

    K8s rejects CR patches over ~1 MiB with HTTP 413. The aggregator
    docstring says confidence grows linearly with cells x metrics x
    percentiles — on big sweeps it dominates. We bound the inlined size:
    parent + children stay (small, structural metadata); confidence is
    served via the disk-backed results sidecar instead.
    """
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "s" / "1234"
    _write_json(sweep_dir / "aggregate.json", {"summary": "small"})
    _write_json(sweep_dir / "children.json", [{"name": "c1"}])
    # ~50 KB confidence payload, well above the test cap below.
    big_confidence = {f"row_{i}": list(range(50)) for i in range(500)}
    _write_json(
        base_dir / "aggregate" / "profile_export_aiperf_aggregate.json", big_confidence
    )

    # Lower the cap to force the drop branch.
    monkeypatch.setattr(
        "aiperf.sweep_controller.main._AGGREGATE_INLINE_MAX_BYTES", 1000
    )
    bundle = _load_aggregate_for_cr(base_dir, "ns", "s", "1234")

    assert "parent" in bundle
    assert "children" in bundle
    assert "confidence" not in bundle, (
        "confidence must be dropped when bundle exceeds inline cap"
    )


def test_load_aggregate_for_cr_keeps_confidence_under_cap(tmp_path: Path):
    """Default cap is generous enough that small confidence payloads stay inlined."""
    base_dir = tmp_path
    sweep_dir = base_dir / "ns" / "sweeps" / "s" / "1234"
    _write_json(sweep_dir / "aggregate.json", {"a": 1})
    _write_json(sweep_dir / "children.json", [{"name": "c1"}])
    _write_json(
        base_dir / "aggregate" / "profile_export_aiperf_aggregate.json", {"small": 1}
    )

    bundle = _load_aggregate_for_cr(base_dir, "ns", "s", "1234")
    assert "confidence" in bundle
