"""Memory probe for AIPerf worker layouts (containers vs forkserver-in-one-pod).

Two entry modes:

  worker        - One fresh Python process: import AIPerf preload + load real
                  HF tokenizers for the target models, dirty a small working
                  set, write a JSON snapshot to /shared/<label>.json, and
                  block. Used as the container's main command in "N-container"
                  pods — N of these run side by side under the pod cgroup.

  forkserver    - One Python process that plays WPM + forkserver at once:
                  import AIPerf preload + load the tokenizers, then fork N
                  children that each write their own snapshot and block. The
                  parent also snapshots. Used as the sole container's main
                  command in the single-container pod.

Snapshots are PSS/RssAnon/etc. read from /proc/self/smaps_rollup. Written
to /shared/<label>.json so the orchestrator (outside the pod) can scrape
them via `kubectl cp` without relying on stdout buffering.

The script itself must not import transformers/tokenizers unconditionally at
module-import time, because the forkserver preload list in the real AIPerf
codebase does not include them. We load them only from within the relevant
mode's entrypoint, mirroring the real worker startup path.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import signal
import sys
import time
from pathlib import Path


_FORKSERVER_PRELOAD = [
    "aiperf.common.bootstrap",
    "aiperf.config",
    "aiperf.common.environment",
    "aiperf.common.logging",
    "aiperf.common.enums",
    "aiperf.common.hooks",
    "aiperf.common.messages",
    "aiperf.common.models",
    "aiperf.common.control_structs",
    "aiperf.common.types",
    "aiperf.plugin",
    "aiperf.plugin.enums",
    "aiperf.common.base_service",
    "aiperf.common.base_component_service",
    "aiperf.common.mixins",
    "aiperf.workers.worker",
    "aiperf.workers.inference_client",
    "aiperf.workers.session_manager",
    "aiperf.credit",
    "aiperf.credit.issuer",
    "aiperf.transports",
    "aiperf.transports.aiohttp_client",
    "aiperf.records.record_processor_service",
    "aiperf.metrics",
    "aiperf.post_processors",
    "pydantic",
    "numpy",
    "zmq",
    "uvloop",
    "orjson",
    "msgspec",
    "rich.console",
    "rich.logging",
    "aiohttp",
    "aiofiles",
    "psutil",
]

SHARED = Path("/shared")


def _kib_from(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.endswith(" kB"):
            out[k] = int(v[:-3])
    return out


def snapshot(pid: int | None = None) -> dict[str, int | str]:
    pid = pid or os.getpid()
    status = _kib_from(Path(f"/proc/{pid}/status"))
    smaps = _kib_from(Path(f"/proc/{pid}/smaps_rollup"))
    return {
        "pid": pid,
        "comm": Path(f"/proc/{pid}/comm").read_text().strip()
        if Path(f"/proc/{pid}/comm").exists()
        else "",
        "VmRSS_kB": status.get("VmRSS", 0),
        "RssAnon_kB": status.get("RssAnon", 0),
        "RssFile_kB": status.get("RssFile", 0),
        "Pss_kB": smaps.get("Pss", 0),
        "Pss_Anon_kB": smaps.get("Pss_Anon", 0),
        "Pss_File_kB": smaps.get("Pss_File", 0),
        "Private_Dirty_kB": smaps.get("Private_Dirty", 0),
        "Private_Clean_kB": smaps.get("Private_Clean", 0),
        "Shared_Clean_kB": smaps.get("Shared_Clean", 0),
        "Shared_Dirty_kB": smaps.get("Shared_Dirty", 0),
    }


def do_imports() -> None:
    for mod in _FORKSERVER_PRELOAD:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 - probe script reports and continues
            print(f"    ! preload failed: {mod}: {e!r}", file=sys.stderr, flush=True)


def load_tokenizers(model_ids: list[str]) -> list[object]:
    """Load HF tokenizers fully — including running an encode so lazy state warms."""
    from transformers import AutoTokenizer

    tokenizers: list[object] = []
    sample = (
        "The quick brown fox jumps over the lazy dog. "
        "Memory profiling one two three four five six seven eight nine ten."
    ) * 20
    for m in model_ids:
        print(f"    loading tokenizer: {m}", file=sys.stderr, flush=True)
        t = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        _ = t.encode(sample)
        _ = t.encode(sample[:500])
        tokenizers.append(t)
        print(f"    loaded: {m}", file=sys.stderr, flush=True)
    return tokenizers


def warmup() -> None:
    """Dirty a small working set like the first seconds of a real worker."""
    try:
        from pydantic import BaseModel

        class M(BaseModel):
            x: int
            y: str

        for i in range(1500):
            M(x=i, y=str(i))
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np

        a = np.zeros((256, 256), dtype=np.float64)
        _ = (a + 1.0).sum()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


def write_snapshot(label: str, extra: dict[str, object] | None = None) -> None:
    snap = snapshot()
    snap["label"] = label
    if extra:
        snap.update(extra)
    SHARED.mkdir(parents=True, exist_ok=True)
    path = SHARED / f"{label}.json"
    path.write_text(json.dumps(snap, indent=2))
    print(f"    wrote snapshot: {path}", file=sys.stderr, flush=True)


def wait_for_signal(path: Path, timeout_s: float = 600.0) -> None:
    """Poll for a coordination file from the orchestrator."""
    start = time.monotonic()
    while not path.exists():
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.2)


def mode_worker(label: str, model_ids: list[str]) -> None:
    """One worker-shaped Python process. Stands in for a single worker container."""
    print(f"[worker:{label}] starting, pid={os.getpid()}", file=sys.stderr, flush=True)
    do_imports()
    load_tokenizers(model_ids)
    warmup()
    gc.collect()
    # Let the kernel settle — refcount/gc churn during import takes a beat
    time.sleep(1.0)

    # Emit a pre-sync snapshot so even a lone container can be measured. The
    # orchestrator writes /shared/GO when all pods are ready; we take a second
    # snapshot then (simultaneous across all pods) and keep alive.
    write_snapshot(f"{label}-pre")
    Path(f"/shared/{label}.ready").touch()

    wait_for_signal(Path("/shared/GO"))
    write_snapshot(label)

    # Stay alive so the orchestrator can also read /sys/fs/cgroup from inside
    # the container while the process is live.
    signal.pause()


def mode_forkserver(prefix: str, n: int, model_ids: list[str]) -> None:
    """Parent: import preload + load tokenizers, fork N children.

    This is the realistic forkserver pattern if the AIPerf preload list were
    extended to instantiate tokenizers in the forkserver. In the real code
    you'd use ``multiprocessing.set_forkserver_preload(...)``; here we use
    plain ``os.fork()`` which has the same CoW semantics (children share
    parent's anon heap until they write).
    """
    print(f"[forkserver-parent] starting, pid={os.getpid()}, N={n}", file=sys.stderr, flush=True)
    do_imports()
    load_tokenizers(model_ids)
    warmup()
    gc.collect()
    time.sleep(1.0)

    child_pids: list[int] = []
    for i in range(n):
        pid = os.fork()
        if pid == 0:
            child_label = f"{prefix}-child-{i:02d}"
            warmup()
            gc.collect()
            time.sleep(0.5)
            write_snapshot(f"{child_label}-pre")
            Path(f"/shared/{child_label}.ready").touch()
            wait_for_signal(Path("/shared/GO"))
            write_snapshot(child_label)
            signal.pause()
        else:
            child_pids.append(pid)

    parent_label = f"{prefix}-parent"
    write_snapshot(f"{parent_label}-pre", extra={"child_pids": child_pids})
    Path(f"/shared/{parent_label}.ready").touch()

    wait_for_signal(Path("/shared/GO"))
    write_snapshot(parent_label, extra={"child_pids": child_pids})

    signal.pause()


def _mp_forkserver_child(prefix: str, i: int) -> None:
    """Body of a child spawned via real multiprocessing.forkserver.

    The child inherits the forkserver's anon heap via CoW, so any tokenizer
    already loaded in the forkserver by ``set_forkserver_preload`` is
    immediately available and shared. We verify by pulling it back out.
    """
    from aiperf_mem_probe_tokenizer_preload import get_preloaded, preloaded_models

    child_label = f"{prefix}-mp-child-{i:02d}"
    available = preloaded_models()
    print(
        f"[{child_label}] pid={os.getpid()} preloaded_models={available}",
        file=sys.stderr,
        flush=True,
    )
    # Actually use one of the preloaded tokenizers so we exercise its pages.
    if available:
        tok = get_preloaded(available[0])
        if tok is not None:
            _ = tok.encode("validation sample " * 16)

    warmup()
    gc.collect()
    time.sleep(0.5)
    write_snapshot(f"{child_label}-pre")
    Path(f"/shared/{child_label}.ready").touch()
    wait_for_signal(Path("/shared/GO"))
    write_snapshot(child_label)
    signal.pause()


def mode_mp_forkserver(prefix: str, n: int, model_ids: list[str]) -> None:
    """Use Python's real multiprocessing.set_forkserver_preload mechanism.

    Environment:
        AIPERF_PRELOAD_TOKENIZERS="comma,separated,model,ids" — read by the
        ``aiperf_mem_probe_tokenizer_preload`` side-effect module at
        forkserver-startup.

    The forkserver helper process is a clean Python that imports the
    modules listed in ``set_forkserver_preload`` at startup. Any tokenizer
    loaded by that import stays in the forkserver's anon memory; every
    child spawned via the forkserver context CoW-shares it.
    """
    import multiprocessing

    # The env var must be set before the forkserver helper is spawned —
    # it's inherited into the helper process's environment at that moment.
    os.environ["AIPERF_PRELOAD_TOKENIZERS"] = ",".join(model_ids)

    # Put the preload module on sys.path (and therefore the forkserver's
    # sys.path, which it inherits at helper-startup). The module name must
    # be importable both in the parent (so main-process snapshots include
    # it) and in the forkserver (so preload runs there).
    sys.path.insert(0, "/shared-preload")

    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload(["aiperf_mem_probe_tokenizer_preload"])

    # Parent state: in this mode the parent does NOT itself load the
    # tokenizers. The forkserver does. The parent stays small; this makes
    # the contrast with the os.fork() variant clean.
    do_imports()
    warmup()
    gc.collect()
    time.sleep(1.0)

    procs: list[multiprocessing.Process] = []
    for i in range(n):
        p = ctx.Process(
            target=_mp_forkserver_child,
            args=(prefix, i),
            name=f"mp-child-{i:02d}",
        )
        p.start()
        procs.append(p)

    # Give the forkserver time to boot and preload the tokenizers, then
    # give each child time to connect and settle.
    time.sleep(3.0)

    parent_label = f"{prefix}-mp-parent"
    write_snapshot(
        f"{parent_label}-pre",
        extra={"child_pids": [p.pid for p in procs]},
    )
    Path(f"/shared/{parent_label}.ready").touch()

    wait_for_signal(Path("/shared/GO"))
    write_snapshot(
        parent_label,
        extra={"child_pids": [p.pid for p in procs]},
    )

    signal.pause()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    w = sub.add_parser("worker")
    w.add_argument("--label", required=True, help="snapshot label")
    w.add_argument("--tokenizers", nargs="+", required=True)

    f = sub.add_parser("forkserver")
    f.add_argument("--prefix", required=True, help="label prefix for parent/children")
    f.add_argument("--n", type=int, required=True, help="number of forked children")
    f.add_argument("--tokenizers", nargs="+", required=True)

    mp = sub.add_parser("mp-forkserver")
    mp.add_argument("--prefix", required=True)
    mp.add_argument("--n", type=int, required=True)
    mp.add_argument("--tokenizers", nargs="+", required=True)

    args = ap.parse_args()

    if args.mode == "worker":
        mode_worker(args.label, args.tokenizers)
    elif args.mode == "forkserver":
        mode_forkserver(args.prefix, args.n, args.tokenizers)
    elif args.mode == "mp-forkserver":
        mode_mp_forkserver(args.prefix, args.n, args.tokenizers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
