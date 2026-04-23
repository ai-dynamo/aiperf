# AIPerf worker-layout memory validation on kind

This harness is the actual experiment behind the memory-savings claim for
running AIPerf workers as `forkserver` children of a single container vs.
the current K8s layout (one container per worker + per record-processor).

It runs two pods in a kind cluster using the **real AIPerf preload list**
from `src/aiperf/common/mp_context.py` and loads the real HuggingFace
tokenizers for `Qwen/Qwen3-0.6B` and `openai/gpt-oss-120b` inside every
worker.

## Run

```bash
# default N=10 children
./run.sh

# larger pod
N=16 ./run.sh

# custom cluster name
CLUSTER_NAME=my-kind N=10 ./run.sh
```

Output goes to `./results/` (per-process JSON snapshots + per-cgroup
`memory.current`) and a side-by-side PSS/cgroup comparison is printed.

## What gets measured

Two pods are scheduled on the same kind node:

| pod | layout |
|---|---|
| `mem-containers` | N sibling containers, each a fresh Python that imports the AIPerf preload and loads both tokenizers. Stands in for today's K8s layout. |
| `mem-forkserver` | One container: parent imports preload + loads both tokenizers, then `os.fork()`s N children that CoW-share everything. Stands in for the hypothetical `workers_as_subprocesses` layout with tokenizers lifted into the preload. |

An init container on each pod warms the `/hf_cache` emptyDir with both
tokenizers before the probe starts, so the file-backed download race is
eliminated from the measurement.

The probes write snapshots to `/shared/*.json` but also block until the
orchestrator writes `/shared/GO` — this guarantees every process takes
its PSS reading simultaneously, while every other process is alive.

## Measured results (linux/amd64, cgroup v2 kind)

Both pods running on one kind worker. Tokenizers: `Qwen/Qwen3-0.6B` +
`openai/gpt-oss-120b`.

| N children | Containers total `memory.current` | Forkserver `memory.current` | Delta (saved) | Per-child saving |
|---:|---:|---:|---:|---:|
|  4 | 1 740 MiB |   635 MiB | **1 105 MiB** | ~276 MiB |
| 10 | 4 351 MiB |   937 MiB | **3 414 MiB** | ~341 MiB |
| 16 | 6 956 MiB | 1 238 MiB | **5 718 MiB** | ~357 MiB |

Asymptotic saving: **~350 MiB per added child**, of which ~50 MiB is the
Python/AIPerf import heap and ~300 MiB is the two tokenizers.

### Extrapolated to cluster scale

| Config | Pod savings | 100 pods |
|---|---:|---:|
|  4 workers +  1 RP  | ~0.9 GiB | ~90 GiB |
|  8 workers +  2 RPs | ~3.3 GiB | ~330 GiB |
| 16 workers +  4 RPs | ~5.7 GiB | ~570 GiB |
| 32 workers +  8 RPs | ~11+ GiB | ~1.1 TiB |

## Honest caveats

1. **This measures peak idle footprint after tokenizer load**, not
   steady-state under live benchmark load. A running worker will dirty
   more CoW-shared pages over time, so the real-world savings will
   erode 10–30% from these numbers.
2. **The forkserver mode here uses `os.fork()` directly**. Python's
   actual `multiprocessing.set_forkserver_preload` spawns a helper
   process from a clean interpreter and imports modules there, not in
   the WPM. For the tokenizer CoW to work in the real code, the
   tokenizer must be instantiated inside the forkserver (e.g. via an
   import-side-effect module in `_FORKSERVER_PRELOAD`). That's
   straightforward but non-zero engineering.
3. **`openai/gpt-oss-120b`'s tokenizer is large** (~150 MiB loaded).
   If AIPerf benchmarks typically use only one model with a smaller
   tokenizer, halve these numbers.
4. **Losing per-worker OOM isolation** is the cost: one worker OOMing
   in forkserver mode takes the pod. These savings are meaningful but
   not free.

## Files

```
tools/mem_validate_kind/
├── Dockerfile                            # aiperf-slim + transformers/tokenizers + probe.py
├── probe.py                              # worker / forkserver mode entrypoints
├── report.py                             # aggregates snapshots + cgroup metrics
├── run.sh                                # end-to-end orchestrator
├── manifests/
│   ├── pod-containers.yaml.tpl           # N-container pod template
│   └── pod-forkserver.yaml.tpl           # 1-container + N-fork pod template
└── results-*/                            # per-N data from prior runs
```

## Prerequisites

* `docker` (tested on 28.x)
* `kind` (tested on v0.29)
* `kubectl`
* A local image tagged `aiperf-slim:amd64-final` (built from
  `deploy/Dockerfile.aiperf-slim`)
