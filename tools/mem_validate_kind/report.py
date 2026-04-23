"""Report helper for run.sh: aggregate per-process snapshots + cgroup metrics.

Prints a side-by-side comparison between the N-container and 1-container+N-fork
modes, using both:

  - sum of per-process PSS from /proc/*/smaps_rollup (the anon-accounting
    proxy; includes CoW sharing within a single cgroup), and

  - each container's /sys/fs/cgroup/memory.current (the kernel's own view
    of what the cgroup is billed; in kind this is a cgroup v2 directory).

The PSS sum equals the cgroup's anonymous-memory bill when all sharers are
in the same cgroup (forkserver pod). For the containers pod we just sum
each container's memory.current directly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read_cgroup_kib(path: Path) -> int:
    """Read a memory.current-style file (bytes) and convert to kB."""
    if not path.exists():
        return 0
    try:
        return int(path.read_text().strip()) // 1024
    except Exception:  # noqa: BLE001
        return 0


def _load_snapshots(d: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(d.glob("*.json")):
        if p.name.endswith("-pre.json"):
            continue
        try:
            out.append(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed to parse {p}: {e!r}")
    return out


def mib(kib: int | float) -> str:
    return f"{kib / 1024:8.1f} MiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--containers-dir", required=True)
    ap.add_argument("--forkserver-dir", required=True)
    ap.add_argument("--mp-forkserver-dir", required=False, default=None)
    ap.add_argument("--n", type=int, required=True)
    args = ap.parse_args()

    containers_dir = Path(args.containers_dir)
    forkserver_dir = Path(args.forkserver_dir)
    mp_dir = Path(args.mp_forkserver_dir) if args.mp_forkserver_dir else None

    c_snaps = _load_snapshots(containers_dir)
    f_snaps = _load_snapshots(forkserver_dir)
    m_snaps = _load_snapshots(mp_dir) if mp_dir and mp_dir.exists() else []

    print()
    print("=" * 78)
    print(f"AIPerf worker-layout memory validation  —  N = {args.n}  —  on kind")
    print("=" * 78)

    # ----- container mode -----
    print()
    print("## CONTAINERS MODE  (N separate sibling containers)")
    print(f"   {len(c_snaps)} worker snapshots loaded")
    if c_snaps:
        print(f"   {'container':<10} {'Pss':>11} {'Pss_Anon':>11} {'Pss_File':>11} "
              f"{'Priv_Dirty':>11} {'RssAnon':>11}")
        for s in c_snaps:
            print(f"   {str(s.get('label', ''))[:10]:<10} "
                  f"{mib(s['Pss_kB']):>11} {mib(s['Pss_Anon_kB']):>11} "
                  f"{mib(s['Pss_File_kB']):>11} {mib(s['Private_Dirty_kB']):>11} "
                  f"{mib(s['RssAnon_kB']):>11}")
        c_pss = sum(s["Pss_kB"] for s in c_snaps)
        c_pss_anon = sum(s["Pss_Anon_kB"] for s in c_snaps)
        c_priv_dirty = sum(s["Private_Dirty_kB"] for s in c_snaps)
        print(f"   {'TOTAL':<10} {mib(c_pss):>11} {mib(c_pss_anon):>11} "
              f"{'-':>11} {mib(c_priv_dirty):>11} "
              f"{mib(sum(s['RssAnon_kB'] for s in c_snaps)):>11}")

    c_cgroup_kib = 0
    for f in sorted(containers_dir.glob("cgroup-*.txt")):
        c_cgroup_kib += _read_cgroup_kib(f)
    print(f"   sum memory.current across {len(list(containers_dir.glob('cgroup-*.txt')))} "
          f"container cgroups: {mib(c_cgroup_kib)}")

    # ----- forkserver (os.fork) mode -----
    print()
    print("## FORKSERVER MODE  (1 container, os.fork N children)")
    print(f"   {len(f_snaps)} parent+child snapshots loaded")
    if f_snaps:
        print(f"   {'process':<14} {'Pss':>11} {'Pss_Anon':>11} {'Pss_File':>11} "
              f"{'Priv_Dirty':>11} {'RssAnon':>11}")
        for s in f_snaps:
            print(f"   {str(s.get('label', ''))[:14]:<14} "
                  f"{mib(s['Pss_kB']):>11} {mib(s['Pss_Anon_kB']):>11} "
                  f"{mib(s['Pss_File_kB']):>11} {mib(s['Private_Dirty_kB']):>11} "
                  f"{mib(s['RssAnon_kB']):>11}")
        f_pss = sum(s["Pss_kB"] for s in f_snaps)
        f_pss_anon = sum(s["Pss_Anon_kB"] for s in f_snaps)
        f_priv_dirty = sum(s["Private_Dirty_kB"] for s in f_snaps)
        print(f"   {'TOTAL':<14} {mib(f_pss):>11} {mib(f_pss_anon):>11} "
              f"{'-':>11} {mib(f_priv_dirty):>11} "
              f"{mib(sum(s['RssAnon_kB'] for s in f_snaps)):>11}")

    f_cgroup_kib = _read_cgroup_kib(forkserver_dir / "cgroup-forkserver.txt")
    print(f"   memory.current of forkserver container cgroup: {mib(f_cgroup_kib)}")

    # ----- mp-forkserver (multiprocessing.set_forkserver_preload) -----
    m_pss = m_pss_anon = m_priv_dirty = m_cgroup_kib = 0
    if m_snaps:
        print()
        print("## MP-FORKSERVER MODE  (multiprocessing.set_forkserver_preload)")
        print(f"   {len(m_snaps)} parent+child snapshots loaded")
        print(f"   {'process':<18} {'Pss':>11} {'Pss_Anon':>11} {'Pss_File':>11} "
              f"{'Priv_Dirty':>11} {'RssAnon':>11}")
        for s in m_snaps:
            print(f"   {str(s.get('label', ''))[:18]:<18} "
                  f"{mib(s['Pss_kB']):>11} {mib(s['Pss_Anon_kB']):>11} "
                  f"{mib(s['Pss_File_kB']):>11} {mib(s['Private_Dirty_kB']):>11} "
                  f"{mib(s['RssAnon_kB']):>11}")
        m_pss = sum(s["Pss_kB"] for s in m_snaps)
        m_pss_anon = sum(s["Pss_Anon_kB"] for s in m_snaps)
        m_priv_dirty = sum(s["Private_Dirty_kB"] for s in m_snaps)
        print(f"   {'TOTAL':<18} {mib(m_pss):>11} {mib(m_pss_anon):>11} "
              f"{'-':>11} {mib(m_priv_dirty):>11} "
              f"{mib(sum(s['RssAnon_kB'] for s in m_snaps)):>11}")
        m_cgroup_kib = _read_cgroup_kib(mp_dir / "cgroup-forkserver.txt") if mp_dir else 0
        print(f"   memory.current of mp-forkserver container cgroup: {mib(m_cgroup_kib)}")

    # ----- deltas -----
    print()
    print("## DELTA (containers − fork-variant)")
    if c_snaps and f_snaps:
        print("  vs forkserver (os.fork):")
        print(f"   sum Pss:            {mib(c_pss - f_pss)}  "
              f"(containers {mib(c_pss)} − forkserver {mib(f_pss)})")
    if c_cgroup_kib and f_cgroup_kib:
        print(f"   cgroup memory.current: "
              f"{mib(c_cgroup_kib - f_cgroup_kib)}  "
              f"(containers {mib(c_cgroup_kib)} − forkserver {mib(f_cgroup_kib)})")

    if c_snaps and m_snaps:
        print("  vs mp-forkserver (multiprocessing.set_forkserver_preload):")
        print(f"   sum Pss:            {mib(c_pss - m_pss)}  "
              f"(containers {mib(c_pss)} − mp {mib(m_pss)})")
    if c_cgroup_kib and m_cgroup_kib:
        print(f"   cgroup memory.current: "
              f"{mib(c_cgroup_kib - m_cgroup_kib)}  "
              f"(containers {mib(c_cgroup_kib)} − mp {mib(m_cgroup_kib)})")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
