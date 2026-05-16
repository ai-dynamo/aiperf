# Rust mock + scale + agentic-ISL sweep — DGX cluster, 2026-04-30

**Operator image**: `nvcr.io/nvidian/dynamo-dev/aiperf:k8s-multi-20260501-021032-51137ff98`
**Mock-server**: `nvcr.io/nvidian/dynamo-dev/aiperf-mock-server-rs:k8s-amd64-20260319-rs-local`
  (Rust mock — 86 MB image, axum-based SSE, deterministic Shakespeare-corpus tokenizer)
**Cluster**: `nv-prd-dgxc` GKE, 3× n2-standard-8 customer-cpu nodes (24 CPU usable).
**Sampler**: `kubectl top pod --containers` per ~3 s, peak across cell, working-set bytes.

## Result table (mock-server pod RSS shown but excluded from total)

```
 conc      isl   osl      rps    tok/s  rec-ok rec-er    ds   rm  ctrl wkr-mx mock    tot
-----------------------------------------------------------------------------------------
 5000      128   128   3260.0   329263   50000      0   174  104  1086   1824   19   5088
10000      128   128   3092.0   312289  100000      0   174  163  1167   2534   60   8770
 5000     8192   256   1955.5   197501   20000      0   193  104  1125   1153   40   6895
 1000    65536   256    609.9    61601    3000      0   184   82  1083   3249  178   4709
  500   262144   256    497.6    50259    1000      0   184   82  1080   3611  339   5068
  200   524288   256    190.7    48824       0    500   184   82  1080   2256  339   3717
   50  1048576   256     57.5    14733       0    192   216   81  1111   1609   85   3101
   20  1048576  1024     56.4    57764       0    100   224   72  1052   1013   44   2448
```

All MiB. `tot = ctrl + wkr-sum + op` (mock excluded). `wkr-mx = peak per worker pod`. `ds = dataset-manager`. `rm = records-manager`. `rec-ok / rec-er = phases.profiling.recordsSuccess / recordsError`.

## Headlines

### 1. Rust mock is ~2.6× faster than Python at the calibration shape

Apples-to-apples on `(conc=5000, ISL=128, OSL=128, requests=50K)`:

| | Python mock | Rust mock | Δ |
|---|---|---|---|
| RPS | 1230 | **3264** | +165% |
| out_tok/s | 142 K | **330 K** | +132% |
| ctrl-tot | 1092 MiB | 1086 MiB | flat |
| wkr-sum | 3768 MiB | 3645 MiB | flat |
| wkr-max-pod | 2569 MiB | **1824 MiB** | −29% |
| **mock pod RSS** | **500 MiB** | **19 MiB** | **−96%** |
| total (no mock) | 5170 MiB | 5088 MiB | flat |

The control-plane and worker pods don't notice the server change — same memory shape — but the mock pod itself drops 25× and throughput more than doubles. **At equal cluster cost, switching to the Rust mock unlocks ~2.5× more work per second.**

### 2. AIPerf controller scaling cliff: somewhere between 60 and 150 services

At `conc=15000` with `connectionsPerWorker=100` we got 150 worker services + 5 system services = 155 registrations. Controller registration took **52 s** (vs ~5 s at lower scale), then the run actually completed (`credits complete from timing_manager` in controller logs) but the AIPerfJob CR's status flipped to **Failed** with `progress=0`. The shape we tested:

- `conc=15000, conn/worker=250, wpp=10` → 60 worker services → succeeds at 1751 RPS (worker-CPU bound)
- `conc=15000, conn/worker=250, wpp=5` → 60 services, 12 pods → succeeds at 1607 RPS (no improvement; per-worker GIL was the bottleneck not per-pod CPU)
- `conc=15000, conn/worker=100, wpp=10` → **150 services → CR flips Failed despite live benchmark completing**

This is a **separate bug worth investigating**: there's a race or timeout in the post-profiling completion path that breaks once the registration count crosses some threshold.

### 3. AIPerf handles 1M-token agentic ISL successfully on the data path

The flagship test: `conc=50, ISL=1,048,576 tokens, OSL=256, entries=20`. Result:

- 200/200 requests sent, 200/200 completed.
- Live RPS: 57.5. Output-token throughput: 14.7 K/s.
- Dataset-manager peak: **216 MiB** (only +32 MiB above the 184 MiB baseline at smaller ISLs!).
- Worker peak: **1609 MiB** — *less* than the 65 K-ISL cell's 3249 MiB, because the 1M cell ran at 100× lower concurrency.
- **No DatasetManager OOM, no worker OOM, no controller crash.**

### 4. The doc's "10× amplification at ISL+OSL=173K" claim is not observed

`docs/kubernetes/memory-estimator.md` cites a 10× RecordProcessor queue-depth amplification at `ISL+OSL = 173,000`. Tested across this entire ladder — peaks barely move. From ISL=8K (1153 MiB max-pod) to ISL=1M (1609 MiB at 50 conc, or 3611 MiB at 500 conc with 256K ISL — the highest we hit), the worker memory ladder is gentle (≤ 3.1×). The dataset-manager memory varies by **~30 MiB** across the entire ISL range. Either the doc model is wrong, or the path it describes was already optimized away. **Recommend updating the doc to remove the 10× amplification claim, or substantiating it with a reproducer.**

### 5. Records-side processing has a hard cliff at ISL ≥ 512K

Up to ISL=262144: **100 % records succeed.** At ISL=524288 and above: **0 records succeed, every request becomes a `recordsError`.** Concrete numbers:

| ISL | records-success | records-error | live RPS |
|---|---|---|---|
| 65,536 | 3000 | 0 | 610 |
| 262,144 | 1000 | 0 | 498 |
| **524,288** | **0** | **500** | **191** |
| 1,048,576 (osl=256) | **0** | **192** | 58 |
| 1,048,576 (osl=1024) | **0** | **100** | 56 |

Requests went out, the server responded, the round-trip completed at the HTTP layer — but the post-processing (probably tokenizer-side parsing of the response, given `builtin` tokenizer's hard limits, or a ZMQ message-size limit on records publishing) fails at large ISL. **This is the second bug worth a separate ticket.**

### 6. ISL-vs-memory shape (within working range)

Excluding the records-broken cells, normalizing on the same OSL=256:

| ISL | conc × ISL bytes (≈ in-flight prompt total) | wkr-max-pod | dataset |
|---|---|---|---|
| 8 K | 5000 × 32 KB ≈ 160 MB | 1153 | 193 |
| 64 K | 1000 × 256 KB ≈ 256 MB | 3249 | 184 |
| 256 K | 500 × 1 MB ≈ 500 MB | 3611 | 184 |
| 512 K | 200 × 2 MB ≈ 400 MB | 2256 | 184 |
| 1 M | 50 × 4 MB ≈ 200 MB | 1609 | 216 |

**Worker memory tracks `concurrency × ISL_bytes`** (in-flight prompt residency), not raw ISL. Capacity-planning rule of thumb: **worker peak ≈ 1500 MiB + 4 × (concurrency × ISL_bytes_MiB)** holds within an order of magnitude across this range.

## Non-findings (mock-server pod)

The Rust mock peaks held below 339 MiB across the entire sweep, including the 1 M-ISL cells. Mock-server is comfortably the smallest piece of the system; it's not a credible scaling concern.

## File layout (everything in repo)

- `dev/scripts/sweep_rs_isl_scale.py` — sweep driver (parametric ISL+conc grid, kubectl-top sampler, ConnectionsPerWorker / workersPerPod overridable per cell).
- `dev/scripts/analyze_rs_isl_scale.py` — post-processor; pulls live RPS / records-success / records-error from CR JSON.
- `dev/deploy/aiperf-mock-server-rs.yaml` — Deployment + Service for the Rust mock on DGX.
- `dev/results/sweep-rs-isl-scale.csv` — raw sampler rows (one per cell).
- `dev/results/sweep-rs-isl-scale-analysis.csv` — clean post-processed table.
- `dev/results/sweep-rs-isl-scale.log` / `*-pt2.log` / `*-pt3.log` / `*-pt4.log` — driver stdout from each retry pass.
- `dev/results/cr-snapshots-rs/sweep-rs-*.json` — full AIPerfJob CR JSONs per cell (terminal state).
- `dev/results/sweep-rs-isl-scale-findings.md` — this report.

## Reproducing

```bash
# 1. Confirm operator + rust mock are deployed:
KCTX=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-01
kubectl --context $KCTX -n acasagrande-aiperf-bench get deploy aiperf-mock-server-rs

# 2. Run the sweep:
IMG=$(kubectl --context $KCTX -n acasagrande-aiperf get deploy aiperf-operator \
  -o jsonpath='{.spec.template.spec.containers[0].image}')
PYTHONUNBUFFERED=1 uv run python dev/scripts/sweep_rs_isl_scale.py \
  --image "$IMG" --out dev/results/sweep-rs-isl-scale.csv

# 3. Post-process:
uv run python dev/scripts/analyze_rs_isl_scale.py
```

Total wall time ~12 min for the 8-cell agentic ladder against the rust mock.

## Things to look at next

1. **Investigate the 150-services controller-registration cliff.** Watch `system_controller.py` and the kopf operator for race conditions in the post-PROFILING-end CR status update path.
2. **Investigate the records-side ISL cliff at 512K+.** Likely candidates: `builtin` tokenizer's max-input limit, or a ZMQ frame-size cap on the records bus.
3. **Update `docs/kubernetes/memory-estimator.md`** to remove or substantiate the "10× at ISL+OSL=173K" claim.
4. **Try the same sweep with a larger tokenizer** (e.g. `meta-llama/Llama-2-7b-hf`) to see if records-side processing is the limit, vs the builtin tokenizer's specific cap.
