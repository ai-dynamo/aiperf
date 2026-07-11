<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: Hash-Derived RNG System (`aiperf-rng`)

**Date:** 2026-07-10
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** design
**Companions:** `2026-07-10-aiperf-rust-coverage-gap-ledger.md` §7 (the
`random_generator.py` determinism substrate + the "cross-language reproducibility?"
decision), `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md`,
`2026-07-10-aiperf-rust-scheduling-policy-sketch.md` (arrival-interval consumer),
`2026-07-10-aiperf-rust-dataset-segment-seam-design.md` (dataset/sampler consumer).
**Memory anchor:** the coverage-gap ledger's decision #2 ("cross-language
byte-for-byte RNG reproducibility: yes or no?"). **Answered here: NO.** See §2.

---

## 0. The one idea (why this is beautiful)

AIPerf never threads a seed through call sites. A component anywhere in the tree
gets its own reproducible, statistically-independent RNG by **naming itself**:

```python
self._rng = rng.derive("dataset.audio.duration")   # that's the entire ceremony
```

The magic is a single line of algebra (`random_generator.py:414-416`):

```python
seed_string = f"{root_seed}:{identifier}"
hash_bytes  = hashlib.sha256(seed_string.encode("utf-8")).digest()
child_seed  = int.from_bytes(hash_bytes[:8], byteorder="big")
```

That one hash buys four properties at once:

1. **Order-independence.** The child seed depends only on `(root, identifier)`,
   not on how many RNGs were derived before it or in what order. Add a new
   component, reorder `__init__`s, parallelize across workers — every *other*
   stream is unchanged. (Contrast a `split()`/counter scheme, where inserting one
   draw upstream shifts everything downstream.)
2. **Hierarchy for free.** Dotted identifiers (`dataset.audio.duration`) are just
   strings; the namespace is documentation, not machinery. No tree of parent RNGs
   to maintain.
3. **Cross-run / cross-process stability.** A cryptographic hash is stable across
   process restarts, `PYTHONHASHSEED`, and machines — unlike `hash()`. The same
   `(root, identifier)` yields the same stream forever.
4. **One knob.** `root_seed = None` ⇒ every derived stream is non-deterministic
   (OS entropy); `root_seed = 42` ⇒ the whole run is reproducible. The
   deterministic/non-deterministic choice is made once, at the root.

The **`HashIdRandomGenerator`** (`../aiperf-graph-ir/.../hash_id_random_generator.py`)
extends the same trick one level deeper for parallel trace synthesis: it re-seeds
*per datum* via `sha256(f"{seed}:{trace_id}:{hash_id}")`, so N workers decoding the
same KV-block `hash_id` produce **identical tokens** with zero coordination. Same
algebra, applied to `(seed, trace_id, hash_id)` instead of `(root, identifier)`.

**This spec ports the *algebra and semantics* — not Python's PRNG bytes.** Why that
is the right line is §2; the short version is that the beauty above is entirely in
the *derivation*, which is language-agnostic, and none of it depends on
reproducing CPython's Mersenne Twister or NumPy's PCG64 stream.

---

## 1. Ground truth (Python source being ported)

Repo-relative to `/home/anthony/nvidia/projects/aiperf/ajc/rust`, plus the
`aiperf-graph-ir` sibling tree for the (canonical) hash-id layer:

| File | What it defines |
|---|---|
| `src/aiperf/common/random_generator.py` | `RandomGenerator` (dual Python+NumPy backend, 20 methods); `_RNGManager` (root→child seed derivation); module `init`/`derive`/`reset` global singleton; `derive_variation_seed`. |
| `../aiperf-graph-ir/src/aiperf/common/hash_id_random_generator.py` | **Canonical** `HashIdRandomGenerator(RandomGenerator)` — NumPy disabled; `from_base_rng` (seed-0-preserving, non-consuming); `set_trace_id`; `reseed_for_hash_id(hash_id, trace_id=None)`. Newer than the `../agentx` copy, which has the buggy falsy-`or` `from_base_rng` and no per-call scope — do not port agentx's. |
| `src/aiperf/config/distributions.py` | 5-way `Distribution` union (Fixed/Normal/LogNormal/Multimodal/Empirical) with `.sample(rng)`. Consumes `random`, `sample_normal`, `sample_positive_normal`. |
| `src/aiperf/common/models/sequence_distribution.py` | `SequenceLengthDistribution.sample` — `random()` + cumsum `searchsorted` + `sample_positive_normal_integer`. |
| `src/aiperf/dataset/generator/audio.py` | `sample_normal`, `numpy_choice`. |
| `src/aiperf/dataset/agentic_code_gen/distributions.py` | NumPy `Generator.lognormal`/`random` (array path). |

The two `random_generator.py` copies (ajc/rust and agentx) have an **identical
public method set** (verified by diff). Port the method *contract* from ajc/rust;
port the hash-id subclass from `aiperf-graph-ir` (the fixed one).

### 1.1 The derivation surfaces (the crown jewel — port the algebra)

```python
# _RNGManager.derive(identifier)                       random_generator.py:412-420
seed = int.from_bytes(sha256(f"{root}:{identifier}".encode()).digest()[:8], "big")

# derive_variation_seed(root, label)                   random_generator.py:524-528
seed = int.from_bytes(sha256(f"{root}:variation:{label}".encode()).digest()[:8], "big")

# HashIdRandomGenerator.reseed_for_hash_id(hash_id, trace_id=None)   graph-ir hash_id_random_generator.py:87-89
scope = self._trace_id if trace_id is None else trace_id     # "" = GLOBAL namespace
seed = int.from_bytes(sha256(f"{seed}:{scope}:{hash_id}".encode()).digest()[:8], "big")
```

All three are the **same primitive**: `u64(hash(utf8(key))[:8], big-endian)`.
Everything else is choosing the `key` string. This is what we port faithfully — it
is the entire reproducibility + order-independence substrate.

**Hash choice — LOCKED to blake3 (§8 Q2).** The Python source hashes with SHA-256;
the Rust port uses **blake3** (already a workspace dep, faster on these short
inputs, no new crate). Only the hash function changes — the algebra (colon-joined
key → first 8 bytes → big-endian u64) is identical. Because no cross-language byte
parity is required (§2), the Rust seeds *deliberately differ* from Python's; blake3
is chosen for speed on the per-`(trace_id, hash_id)` reseed path (§5). (The `init()`
32-bit numpy-seed fold, `(seed ^ (seed>>32)) & 0xFFFFFFFF`, is a NumPy-seeding
detail that dies with the "match NumPy's stream" goal — see §2/§7.)

### 1.2 Consumer census (namespace + method demand)

**~32 `derive()` identifiers** across the tree (the full namespace to preserve as
string constants, not invent):

```
composer.conversation.turn_count      dataset.prompt.corpus            timing.ramp.poisson
composer.conversation.turn_delay      dataset.prompt.length            timing.request.cancellation
composer.turn.max_tokens              dataset.prompt.prefix            timing.request.gamma_interval
composer.turn.model_selection         dataset.rankings.passages        timing.request.poisson_interval
dataset.audio.data                    dataset.rankings.passages.tokens models.sequence.distribution
dataset.audio.duration                dataset.rankings.query.tokens
dataset.audio.format                  dataset.sampler.random
dataset.image.dimensions              dataset.sampler.shuffle
dataset.image.format                  dataset.synthesis.empirical_sampler
dataset.image.noise                   dataset.synthesis.synthesizer
dataset.image.source                  dataset.video.audio
dataset.loader.random_pool            dataset.video.noise
dataset.loader.sharegpt
```

**Methods actually called** (count across tree — the port surface, priority
order). The "Distribution / semantics" column is what must match Python — *not*
the generator that produces it:

| Method | Uses | Distribution / semantics to preserve |
|---|---|---|
| `sample_positive_normal_integer(mean, stddev)` | 12 | truncated-normal ≥0 then `max(1, ceil(...))`; `stddev<=0 ⇒ max(1, round(mean))` |
| `choice(seq)` | 11 | uniform over a sequence |
| `random()` | 6 | uniform `[0, 1)` |
| `choices(pop, k)` | 6 | uniform with replacement |
| `integers(low, high, size)` | 5 | uniform ints over a range |
| `shuffle(x)` | 4 | uniform permutation (in place) |
| `numpy_choice(a, size, p, replace)` | 4 | weighted choice |
| `expovariate(lambd)` | 3 | Exponential(λ) |
| `randrange`, `randint` | 4 | uniform int over a (half-)open range |
| `gammavariate(alpha, beta)` | 2 | **Gamma(α, β), mean = α·β** (arrival burstiness) |
| `uniform(a, b)` | 1 | uniform `[a, b]` |
| `sample_normal(mean, sd, lo, hi)` | 1 | truncated normal by rejection (≤10 000 iters, clamp fallback) |
| `sample(pop, k)` | 1 | uniform without replacement |
| `random_batch(size)` | 1 | array of uniforms |
| `normal(loc, scale, size)` | 1 | Normal(loc, scale) |

`sample_positive_normal_integer` is >2× the next consumer — it is the hot method
(ISL/OSL sizing). `HashIdRandomGenerator` uses only the scalar-style methods (it
never touches the array ops), so its generator needs no array support.

> Escape hatch seen in the wild: `session_synthesizer.py` calls `self._rng.bytes(16)`
> on a *raw* `random.Random`. Not part of the `RandomGenerator` contract; if a Rust
> consumer needs it, expose `fill_bytes(&mut [u8])` on the generator (any PRNG has
> it). Flagged, not in the v1 surface.

---

## 2. The fidelity ruling (settled)

**Decision (Tech Lead): AIPerf-Rust does NOT need to reproduce the Python tool
byte-for-byte — anywhere.** No regression canary compares Rust output to a Python
reference; no Python-generated reference fixtures are reused; the two tools are not
A/B'd draw-for-draw. This answers coverage-gap ledger decision #2 with a **no**,
and it collapses the design.

With cross-language byte parity off the table, the requirements are exactly three —
**none of which imply matching CPython's or NumPy's PRNG streams:**

1. **Internal reproducibility.** Same `root_seed` ⇒ same run, bit-identical, every
   time, on the same build. (This is what makes a benchmark result quotable and a
   bug reproducible.)
2. **Order-independence.** The derive property from §0 — components are
   statistically independent and stable under reordering, insertion, and
   parallelism. This lives entirely in the **seed derivation**, not the generator.
3. **Distributional + semantic parity with Python.** The *distributions* the load
   generator emits (truncated Normal for ISL/OSL, LogNormal, **Gamma arrivals**,
   weighted Empirical/Multimodal picks) and the *edge-case semantics* (the
   rejection cap + clamp-to-mean fallback, `max(1, ceil(...))`, the `stddev<=0`
   shortcut, `gammavariate` mean = α·β, the weighted-cumulative pick) must match —
   because **those shape the benchmark's realism; the underlying bytes do not.**

### What this kills, and what "algorithm parity" actually means

The earlier draft of this spec over-rotated: it mandated a byte-exact **MT19937**
scalar port and a bespoke **PCG64 + ziggurat** array port. Both existed *only* to
reproduce Python's exact bit stream — which requirement (1) does not need (any
deterministic PRNG gives internal reproducibility) and requirement (3) does not
need either (a *correct* Gamma is a correct Gamma regardless of the sampler). So:

- **Cut:** MT19937, bespoke PCG64, ziggurat tables, CPython's stateful `gauss`
  second-value cache, `genrand_res53`, the NumPy 32-bit seed fold, and the entire
  Python-vs-Rust golden-stream corpus. All were byte-parity machinery.
- **Kept, correctly reframed:** "algorithm parity" means the **distribution
  family and its parameterization match** — Gamma(α, β) with mean α·β, truncated
  normal with the exact rejection/clamp semantics, weighted picks with the exact
  cumulative-walk tie-break. It does **not** mean "reimplement CPython's Gamma
  sampler." `rand_distr::Gamma` (Marsaglia–Tsang) and Python's (Cheng) draw the
  same Gamma(α, β); that is the parity that matters.

### The resulting design (one PRNG + `rand_distr`)

`RandomGenerator` holds **one** deterministic PRNG seeded from the derived `u64`,
and uses `rand` / `rand_distr` for the distributions, with a faithful port of the
thin *wrapper semantics* (rejection loop, `ceil`/`round` boundaries, the
`stddev<=0` shortcuts, the distribution control flow). Recommended PRNG:
`rand_pcg::Pcg64` or `rand::rngs::StdRng` (ChaCha12) — pick one, document it, and
it becomes part of the "same build ⇒ same bytes" internal-reproducibility
contract. No dual backend; no `_python_rng`/`_numpy_rng` split (that mattered only
because Python routed to two *stream-distinct* generators — irrelevant now).

This is strictly less code than the prior draft, and it satisfies every real
requirement. The only thing given up — reproducing Python's exact draws — is the
thing the Tech Lead just confirmed is not needed.

---

## 3. Rust API design (`aiperf-rng`, a new leaf crate)

A new leaf crate, dependency-direction sibling to `aiperf-clock`: depends only on a
hash impl + `rand`/`rand_distr`; **nothing in the workspace depends the other way**.
Consumers (`dataset`, the scheduler in `run.rs`, graph synthesis) depend on it.

```
crates/aiperf-rng/
  Cargo.toml            # blake3 (workspace dep); rand; rand_distr; rand_pcg; getrandom
  src/
    lib.rs
    derive.rs           # the hash algebra (derive / variation / hash-id key) — the substrate
    generator.rs        # RandomGenerator: one PRNG + rand_distr, the 20-method contract
    hash_id.rs          # HashIdRandomGenerator: reseed-per-(trace_id, hash_id)
    dist.rs             # the 5-way SamplingDistribution + SequenceLengthDistribution
```

### 3.1 The derivation seam (`derive.rs`) — the substrate

No global mutable singleton (that was a Python-multiprocess convenience). Pass the
root explicitly as a small `Copy` value; deriving is a pure function.

```rust
/// The root of a reproducible run. `None` ⇒ every derived stream is
/// non-deterministic (OS entropy). Cheap to copy; pass by value.
#[derive(Copy, Clone, Debug)]
pub struct RngRoot(pub Option<u64>);

impl RngRoot {
    /// Port of `_RNGManager.derive`: seed = u64_be(hash(format!("{root}:{id}"))[..8]).
    pub fn derive(self, identifier: &str) -> RandomGenerator { /* … */ }

    /// Port of `derive_variation_seed` (adaptive-sweep overflow path).
    pub fn derive_variation_seed(self, label: &str) -> Option<u64> { /* … */ }
}

/// The shared primitive: u64 from the first 8 bytes of the blake3 digest,
/// big-endian. This one fn is the whole order-independent-derivation contract.
/// (On the hot reseed path, prefer the alloc-free streaming form in §5 —
/// same digest, no `format!` heap allocation.)
pub fn derive_seed_u64(key: &str) -> u64 {
    let d = blake3::hash(key.as_bytes());          // LOCKED: blake3 (§8 Q2)
    u64::from_be_bytes(d.as_bytes()[..8].try_into().unwrap())
}
```

> Ergonomic note: Python's module-global `init/derive/reset` exists so
> `rng.derive(...)` works from any `__init__` without plumbing. In Rust, thread the
> `RngRoot` through construction (it is `Copy`), or hold one `RngRoot` in the run
> context and hand it to each subsystem builder. Do **not** resurrect a global
> `static mut` — it is a GIL-era convenience, not a feature. The `init()` side
> effect of defensively re-seeding the *global* `random`/`numpy` state is **thrown
> away**: Rust has no ambient global RNG to protect.

### 3.2 The generator (`generator.rs`) — one PRNG + `rand_distr`

```rust
/// One deterministic PRNG + rand_distr. Mirrors the RandomGenerator *contract*
/// (same 20 methods, same distribution semantics) — NOT CPython's dual backend.
pub struct RandomGenerator {
    seed: Option<u64>,
    rng: Pcg64,                 // or StdRng; the chosen deterministic PRNG (§2)
}

impl RandomGenerator {
    pub fn from_seed(seed: Option<u64>) -> Self { /* seed the PRNG, or entropy if None */ }
    pub fn reseed(&mut self, seed: u64) { self.rng = Pcg64::seed_from_u64(seed); }
    pub fn seed(&self) -> Option<u64> { self.seed }

    // uniforms / picks → rng + rand::seq:
    pub fn random(&mut self) -> f64;                                   // [0,1)
    pub fn randrange(&mut self, /* start, stop, step */) -> i64;
    pub fn randint(&mut self, a: i64, b: i64) -> i64;                  // inclusive
    pub fn randrange_u64(&mut self, lo: u64, hi: u64) -> u64;          // hash-id fallback draw
    pub fn uniform(&mut self, a: f64, b: f64) -> f64;
    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T;
    pub fn choices<T: Clone>(&mut self, pop: &[T], k: usize) -> Vec<T>;
    pub fn sample<T: Clone>(&mut self, pop: &[T], k: usize) -> Vec<T>;
    pub fn shuffle<T>(&mut self, x: &mut [T]);
    pub fn weighted_choice<T: Clone>(&mut self, a: &[T], p: Option<&[f64]>) -> T;

    // distributions → rand_distr (same family/params as Python; not same bytes):
    pub fn expovariate(&mut self, lambd: f64) -> f64;                 // Exp(lambd)
    pub fn gammavariate(&mut self, alpha: f64, beta: f64) -> f64;     // Gamma, mean = alpha*beta
    pub fn normal(&mut self, loc: f64, scale: f64) -> f64;            // Normal(loc, scale)

    // bounded-normal wrappers — port the SEMANTICS exactly (§4):
    pub fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> f64;
    pub fn sample_positive_normal(&mut self, mean: f64, stddev: f64) -> f64;   // lower = 0
    pub fn sample_positive_normal_integer(&mut self, mean: f64, stddev: f64) -> i64;

    // array conveniences (same distributions, just batched):
    pub fn integers(&mut self, low: i64, high: Option<i64>, size: usize) -> Vec<i64>;
    pub fn normal_batch(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64>;
    pub fn random_batch(&mut self, size: usize) -> Vec<f64>;
}
```

There is no `Mt19937`/`Pcg64` *pair* and no `gauss_next` state — `rand_distr::Normal`
is stateless from the caller's view, which is fine because we are not preserving
CPython's paired-draw stream.

### 3.3 The hash-id layer (`hash_id.rs`) — the piece you asked for

Port the **canonical** `aiperf-graph-ir` version (it fixed the seed-0 bug and added
a per-call scope override). The generator underneath is the single PRNG; the
hash-id logic is pure derivation + reseed, so it is language-agnostic.

```rust
/// Re-seeds deterministically per (trace_id, hash_id) so parallel workers decoding
/// the same KV block produce identical tokens, order-independent. Uses only the
/// scalar-style methods (no array ops).
pub struct HashIdRandomGenerator {
    seed: u64,               // base seed; appears in the sha256 key below
    trace_id: String,        // instance scope; "" = GLOBAL namespace
    gen: RandomGenerator,
}

impl HashIdRandomGenerator {
    /// `from_base_rng` (aiperf-graph-ir `hash_id_random_generator.py:53-56`):
    ///   base_seed = base.seed  IF present  ELSE base.randrange(0, 2^64)
    /// CRITICAL: read `.seed` WITHOUT consuming base state; seed 0 is LEGAL and is
    /// preserved (only `None`/seedless draws a fallback). The old agentx `or` form
    /// treated 0 as falsy and silently drew — do NOT replicate that.
    pub fn from_base(base: &mut RandomGenerator) -> Self {
        let seed = match base.seed() {
            Some(s) => s,                          // includes Some(0)
            None => base.randrange_u64(0, u64::MAX /* 2^64 */),
        };
        Self { seed, trace_id: String::new(), gen: RandomGenerator::from_seed(Some(seed)) }
    }

    pub fn set_trace_id(&mut self, trace_id: impl Into<String>) { self.trace_id = trace_id.into(); }

    /// reseed with u64_be(hash(f"{seed}:{scope}:{hash_id}")[:8]). `trace_id: Some(_)`
    /// overrides the instance scope for THIS call WITHOUT mutating it (a shared
    /// generator serves interleaved trace scopes); `None` uses the instance
    /// `trace_id` ("" ⇒ GLOBAL namespace, the linear-mooncake / content-global path).
    pub fn reseed_for_hash_id(&mut self, hash_id: i64, trace_id: Option<&str>) {
        let scope = trace_id.unwrap_or(&self.trace_id);
        self.gen.reseed(derive_seed_u64(&format!("{}:{}:{}", self.seed, scope, hash_id)));
    }
    // then: choice/randrange/random/... via Deref to the inner generator.
}
```

Four cases the graph-ir unit tests (`test_hash_id_random_generator.py`) pin — these
are **self-consistency** contracts (Rust-vs-Rust), not Python-golden:

1. `from_base(seed=0)` ⇒ `gen.seed() == Some(0)` (seed 0 preserved).
2. `from_base(seed=0)` does **not** consume the base RNG's state (reading `.seed`
   is side-effect-free; a fresh base draws the same next value).
3. two `from_base(seed=0)` + `reseed_for_hash_id(7, Some("trace"))` produce
   **identical** `randrange(0, 10^6)` sequences (determinism + order-independence).
4. `from_base(None)` (seedless) ⇒ `gen.seed()` is `Some(_)` (a fallback was drawn).

---

## 4. Semantic invariants to preserve (the parts that shape the benchmark)

These are *not* about matching Python's bytes — they are the distribution
semantics and edge cases that change the workload if you get them wrong. Port the
control flow exactly; draw the underlying randomness from `rand`/`rand_distr`.

- **`sample_normal` rejection loop** (`random_generator.py:271-290`): validate
  `lower <= upper` (error otherwise); up to **10 000** `normal(mean, stddev)` draws,
  return the first in `[lower, upper]`; on exhaustion **warn and return
  `clamp(mean, lower, upper)`**. The cap and fallback are the *behavior* (they
  bound latency and define the degenerate-bounds result) — keep both. (The *number*
  of draws no longer needs to match Python, since streams differ; the cap and the
  fallback value do.)
- **`sample_positive_normal_integer`** (`:312-314`): `stddev <= 0 ⇒ max(1,
  round(mean))`; else `max(1, ceil(sample_positive_normal(mean, stddev)))`. `ceil`
  (not `round`) so a sub-1 sample still yields ≥1. Match Python's banker's-rounding
  `round()` for the shortcut so config-time expectations line up. Hot path (ISL/OSL).
- **`gammavariate` parameterization**: Gamma with **mean = α·β**; for arrival
  intervals at rate `r` with smoothness `s`, `interval = gammavariate(s, 1/(r·s))`.
  Use `rand_distr::Gamma` — the *distribution* must match, the sampler need not.
- **Distribution control flow** (`distributions.py`, `sequence_distribution.py`):
  port verbatim — the weighted pick `r = random() * total; walk cumulative`, and
  `searchsorted(cumsum, r, side="right")` then `min(idx, n-1)` clamp. The 5-way
  discriminator (Fixed/Normal/LogNormal/Multimodal/Empirical), the LogNormal
  `sigma = sqrt(2·ln(mean/median))` / `mu = ln(median)` derivation, and the `min`/
  `max` bound-clamping in `Distribution.sample` are pure logic — keep exactly.
- **Seedless (`root_seed = None`)**: every derived generator seeds from OS entropy
  (`getrandom`), so runs are non-reproducible by construction — matching Python's
  `None` pass-through.

---

## 5. Performance & integrity (they do not trade off here)

The RNG is on the hottest paths — ISL/OSL sizing per request, arrival-interval
generation per dispatch, and per-`(trace_id, hash_id)` reseeds during synthesis
(the `transport_bench` >1M req/s regime). The key point: **speed and determinism
are the same design here, not a trade-off.** The order-independence that gives
reproducibility (§0) also gives lock-free per-thread independence.

- **Derive once, at construction — never per draw.** `derive()` is a hash over a
  short string; call it in the component's constructor and keep the
  `RandomGenerator` (as Python does in `__init__`). Steady-state per-draw cost is
  then just the PRNG + distribution, with no hashing.
- **PCG64 on the hot path.** `rand_pcg::Pcg64` is a handful of ALU ops per draw and
  inlines cleanly; prefer it to `StdRng` (ChaCha12), whose cryptographic strength
  buys a benchmark nothing. It is fully deterministic under a fixed seed — the speed
  costs zero integrity.
- **Per-thread, lock-free by construction.** Because each component/worker derives
  its *own* generator from its identifier, there is no shared RNG, no `Arc`, no
  `Mutex`, nothing to contend — matching the repo's "no `Arc`/`Mutex` on hot paths"
  rule. On the thread-per-core runtime the generator is plain owned `&mut` state
  (not even `Rc`/`RefCell`). Reproducibility and zero-synchronization are the same
  property.
- **Alloc-free reseed on the synthesis path.** The naive
  `format!("{seed}:{scope}:{hash_id}")` heap-allocates a `String` per reseed — and
  per-request alloc churn is this repo's top profiled hotspot (why mimalloc is
  global). Feed the hasher the *same bytes* incrementally instead: decimal-ASCII of
  `seed` (stack `itoa` buffer) · `b":"` · `scope.as_bytes()` · `b":"` ·
  decimal-ASCII of `hash_id`. **Identical digest, zero heap** — integrity preserved
  (same key bytes ⇒ same seed, still matches the §6 vectors), allocation eliminated.
- **blake3 is the hash (locked), and reseed is where it earns it.** `derive()` is
  amortized at construction, but `reseed_for_hash_id` can fire per KV block. blake3
  (already a workspace dep) is markedly faster than SHA-256 on these ~30-byte
  inputs, so it is the chosen hash (§8 Q2) — combined with the alloc-free streaming
  key above, a reseed is one small hash over a stack buffer, no heap, no
  crypto-grade overhead.
- **Batch the array ops.** `random_batch`/`integers`/`normal_batch` construct the
  `rand_distr` sampler once and fill a slice, not once per element.

**Integrity guardrails perf must not cross:** never seed a hot generator from
wall-clock/entropy to "go faster" (kills reproducibility); never share one
generator across threads instead of deriving separate ones (kills order-independence
*and* adds contention); never swap the hash or PRNG mid-project without treating it
as a canary-breaking reseed of the world. Every speed lever above — PCG64,
alloc-free hashing, per-thread independence, batching — is orthogonal to
determinism.

---

## 6. Testing (Rust-vs-Rust — no Python oracle)

Because there is no cross-language contract, the suite pins *internal* properties,
not Python bytes.

- **Derivation determinism + order-independence.** `derive` is a pure function of
  `(root, identifier)`; pin a few values as regression fixtures (these are just
  "what our hash produces", stable per build):

  ```
  # blake3(f"{root}:{id}") first 8 bytes, big-endian (LOCKED, §8 Q2).
  # NOT equal to Python's sha256-derived seeds — intentional (no cross-lang parity).
  derive(42, "dataset.loader")               = 2466643113772406410
  derive(42, "timing.request_rate")          = 12613212627144784801
  derive(42, "")                             = 1788878741536589501
  derive(0,  "a")                            = 10339543760652402899
  derive_variation_seed(42, "concurrency=4") = 10717291070465836476
  ```

  Property test: for any permutation of a set of identifiers, each derived seed is
  unchanged (order-independence); distinct identifiers give distinct seeds.
- **Reproducibility.** Same `root_seed` ⇒ identical draw sequences from every
  derived generator and from `HashIdRandomGenerator` after `reseed_for_hash_id`
  (the four §3.3 cases). Different seeds ⇒ different sequences.
- **Distributional correctness.** For `sample_normal`/`sample_positive_normal_integer`/
  `gammavariate`/`expovariate`/the 5-way distributions: statistical checks (sample
  mean/variance within tolerance of the analytic values; truncation bounds
  respected; `gammavariate(s, 1/(r·s))` has mean `1/r`). Port the Python *unit*
  tests (`test_random_generator.py`, `test_distributions.py`,
  `test_sequence_distribution.py`) as behavioral assertions, dropping any that
  assert exact Python draw values.
- **Regression canary (Rust-internal).** Mirror the *concept* of
  `test_random_generator_canary.py`, not its reference file: run a fixed
  `--random-seed 42` profile through the Rust dataset path, snapshot the generated
  inputs once, commit it, and assert future runs reproduce it byte-for-byte. This
  catches accidental determinism breakage (a reordered draw, a changed PRNG) — the
  same protection the Python canary gives, but against a Rust baseline.

---

## 7. Scope boundaries

- **In:** the `RngRoot` derivation seam (`derive` / `derive_variation_seed`), the
  single-PRNG `RandomGenerator` (20-method contract over `rand` + `rand_distr`),
  `HashIdRandomGenerator` (seed-0-preserving, non-consuming, per-call scope), the
  5-way `SamplingDistribution` + `SequenceLengthDistribution`, and the §4 semantic
  invariants. This is the whole "derive system + hash rng" the request names.
- **Thrown away (byte-parity machinery — unneeded per §2):** MT19937, bespoke
  PCG64, ziggurat tables, CPython's stateful `gauss` cache, `genrand_res53`, the
  NumPy 32-bit seed fold, and any Python-vs-Rust golden-stream corpus. Also gone:
  the module-global `init/derive/reset` singleton (→ explicit `RngRoot`), the
  defensive global `random.seed`/`np.random.seed` (no ambient Rust RNG), and the
  runtime `_DisabledNumpyRNG` guard (the array/scalar split disappears with the
  dual backend).
- **Not a separate concern:** the array-vs-scalar distinction. Python split
  because it held two stream-distinct generators; one PRNG makes `integers`/
  `shuffle`/`normal_batch` just batched calls on the same generator.

---

## 8. Open questions

1. **Fidelity ruling — RESOLVED (§2):** no cross-language byte parity anywhere;
   the requirements are internal reproducibility + order-independence +
   distributional/semantic parity, all met by one deterministic PRNG + `rand_distr`
   + a faithful semantic port. No MT19937, no bespoke PCG64.
2. **Hash choice — RESOLVED: blake3.** `derive_seed_u64` uses **`blake3`** (already
   a workspace dep, faster than SHA-256 on the short reseed keys, no new crate). The
   seed need not match Python (§2), so the loss of the incidental SHA-256/Python
   match costs nothing. This is now part of the internal-reproducibility contract:
   the §6 vectors are blake3, and changing the hash later reshuffles every seed and
   breaks the canary — so it is pinned, not revisited.
3. **PRNG choice — pin it.** `rand_pcg::Pcg64` (small, fast, jumpable) vs
   `rand::rngs::StdRng` (ChaCha12, cryptographic-grade, `rand`'s default). Either
   satisfies internal reproducibility; document the pick and treat a change as a
   canary-breaking event. Lean `Pcg64` for speed on the hot ISL/OSL path.
4. **`RngRoot` plumbing ergonomics.** Explicit threading vs a per-thread
   `thread_local!` holding the run's `RngRoot` (set once at thread-per-core startup,
   read by `derive`). The thread-local recovers Python's "call `derive` from
   anywhere" ergonomics without a mutable global. Lean thread-local, set once, never
   reset mid-run.
