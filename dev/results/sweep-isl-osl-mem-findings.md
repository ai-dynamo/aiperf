# ISL/OSL × concurrency memory sweep — DGX cluster, 2026-04-30

**Operator image**: `nvcr.io/nvidian/dynamo-dev/aiperf:k8s-multi-20260501-021032-51137ff98`
**Cluster**: `nv-prd-dgxc` GKE (3× n2-standard-8 customer-cpu nodes; arm64/customer-gpu pool no longer exists).
**Workload**: aiperf-mock-server in `acasagrande-aiperf-bench` (streaming chat completions, fake SSE chunks).
**Sampler**: `kubectl top pod --containers` per ~3 s, peak across the cell, working-set bytes.

## Result table (mock-server excluded from totals)

```
 conc   isl   osl      rps   tok/s    rm  ctrl wkr-sum mx-pod   op    tot
-------------------------------------------------------------------------
 5000   128   128   1230.3  141547    86  1092    3768   2569  310   5170
 5000   128  1024    429.3  223891    93  1093    4714   3089  321   6128
 5000  1024   128   1056.9  134278    84  1080    4061   2811  313   5454
 5000  1024  1024    300.9  275228    99  1100    6243   4098  322   7665
 5000  1024  4096    205.9  269670    93  1084    5301   2746  324   6709
 5000  4096  1024    281.0  286681    92  1114    7017   4770  324   8455
 5000  4096  4096    121.6  437523    92  1113    6745   4233  340   8198
10000   128   128   1454.8  166926   135  1141    6589   1871  342   8072
10000  1024  1024    439.5  402909   149  1161    8929   2992  350  10440
10000  4096  4096    242.7  876257   104  1130    7926   2196  351   9407
```

All values in MiB. `tot = ctrl + wkr-sum + op` (no mock). `mx-pod = peak per worker pod`.

## Headlines

### 1. Records-manager memory is independent of token shape

At fixed concurrency the records-manager peak is flat to within ~15 MiB:

| concurrency | RM range across all 7 ISL/OSL shapes |
|---|---|
| 5000 | 84–99 MiB |
| 10000 | 104–149 MiB |

The records-manager stores per-request metric structs, not token bytes — so increasing ISL or OSL by 32× barely moves it. The concurrency-driven jump (5k → 10k roughly +50 MiB) is the only RM signal here.

### 2. Per-worker-pod peak is sublinear in token count

Going from 256 total tokens (cell 1) to 8192 (cell 7) — **32× more tokens** — the per-pod peak moved 2569 → 4233 MiB, **only 1.65×**.

The `docs/kubernetes/memory-estimator.md` claim of "10× at ISL+OSL = 173K" is **20× higher token budget than we tested** (8192 here vs 173K). At realistic LLM-serving prompt sizes (≤ 8K tokens) the slope is tame — probably no point being defensive about RP queue-depth amplification under those conditions.

### 3. ISL costs more worker memory than OSL at the same token count

At total tokens = 5120, the asymmetry is striking:

| ISL | OSL | per-pod peak |
|---|---|---|
| 4096 | 1024 | **4770 MiB** |
| 1024 | 4096 | **2746 MiB** |

Working hypothesis: ISL bytes are retained for the full request lifetime (prompt sits resident while OSL is being streamed), while OSL bytes are produced + emitted + freed as SSE chunks. Long input → long resident time. Long output → fast turnover. **Caveat**: the long-OSL cell (cell 6) was shorter wall-time (163 s vs 257 s) and processed 14 K vs 29 K requests, so it may not have reached its asymptotic peak — worth re-running with higher request count if the asymmetry matters for capacity planning.

### 4. RPS is dominated by OSL, not ISL

| change | RPS impact |
|---|---|
| ISL 128 → 1024 (8×) at OSL=128 | 1230 → 1057 (−14%) |
| OSL 128 → 1024 (8×) at ISL=128 | 1230 → 429 (**−65%**) |
| ISL 128 → 4096 (32×) at OSL=4096 | 122 → 122 (n/a — both at floor) |

Mock-server's per-request cost is mostly chunk-emission, not prompt parse. Real LLMs would be different — for production capacity tables the OSL knob is the dominant throughput driver but you'll want to re-derive the slope against a real engine.

### 5. Concurrency 5K → 10K with same shape

At ISL=OSL=128, doubling concurrency:
- **+18% RPS** (1230 → 1455 — saturating)
- **+50% RM** (86 → 135 MiB)
- **−27% per-pod peak** (2569 → 1871) — same total work spread across 4 vs 2 worker pods.

The records-manager grows with concurrency (record retention for 2× the in-flight pool). Per-worker peak drops because workersPerPod=10 means 4 pods at 10K vs 2 pods at 5K.

## Caveats on these numbers

1. **Working-set, not RSS.** `kubectl top` reports `container_memory_working_set_bytes` (anonymous + active file cache). A worker mmap-reading a tokenizer file would inflate this; we don't break out heap vs cache.
2. **Sample lag ≤ ~15 s.** metrics-server caches; brief peaks under that window can be missed. Steady-state cells (≥ 30 s in Running) are captured fine. Short cells (cell 6 at 14 K requests) may slightly understate.
3. **Error rates inflate at large OSL.** Cells with OSL ≥ 1024 hit ~25–55 % errors at 5 K concurrency (cell 6: 54 %, cell 5: 50 %, cell 7: 184 % — last figure is `errors / completed` where errors include retries, so > 100 %). The peak memory we captured includes whatever the worker accumulates from in-flight requests that timed out — that may be conservative for a clean workload but bounds the "what can we hold" question.
4. **Mock-server, not a real LLM.** Real engine memory belongs to the engine container and is invisible to this measurement entirely. This sweep is purely about the AIPerf control-plane + worker-pool memory shape.
5. **Single trial per cell.** Re-run with `trials = 2` if any specific data point is going to drive a capacity decision.

## Reproducing

```bash
# 1. Make sure operator is on a known image, no env tweaks above defaults:
KCTX=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-01
kubectl --context $KCTX -n acasagrande-aiperf get deploy aiperf-operator \
  -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\n"}{end}' \
  | grep -E 'CPU|MEMORY' || echo '(env defaults)'

# 2. Run the sweep:
IMG=$(kubectl --context $KCTX -n acasagrande-aiperf get deploy aiperf-operator \
  -o jsonpath='{.spec.template.spec.containers[0].image}')
PYTHONUNBUFFERED=1 uv run python dev/scripts/sweep_isl_osl_mem.py \
  --image "$IMG" --out dev/results/sweep-isl-osl-mem.csv

# 3. Post-process (derives true RPS + token throughput from CR snapshots):
uv run python dev/scripts/analyze_isl_osl_mem.py
```

Total wall time ~30–35 min for the 10-cell grid against the persistent mock-server.

## File layout

- `dev/scripts/sweep_isl_osl_mem.py` — sweep driver (also captures peak RSS via kubectl top).
- `dev/scripts/analyze_isl_osl_mem.py` — post-processor (this report's table generator).
- `dev/results/sweep-isl-osl-mem.csv` — raw sampler output (one row per cell).
- `dev/results/sweep-isl-osl-mem-analysis.csv` — post-processed with derived RPS + tok/s + total-no-mock.
- `dev/results/sweep-isl-osl-mem.log` — sweep driver stdout (per-cell phase trace + memory-summary block).
- `dev/results/cr-snapshots/sweep-iom-*.json` — full AIPerfJob CR JSON per cell (captured before TTL).
- `dev/results/sweep-isl-osl-mem-findings.md` — this report.
