<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Random Number Generation & Reproducibility

**Quick Links:**<br>
[Overview](#overview) • [What Is Reproducible](#what-is-reproducible-what-is-not) • [User Guide](#user-guide) • [Developer Guide](#developer-guide) • [Reference](#reference)

---

## Overview

**TL;DR:** Use `--random-seed 42` to get **identical dataset content** across runs. Performance metrics and worker assignment vary due to distributed system architecture.

AIPerf provides **deterministic reproducibility** for all seed-controlled randomness using hash-based RNG derivation. This enables reproducible dataset generation while maintaining realistic load testing performance.

> [!IMPORTANT]
> **Default behavior:** Without `--random-seed`, AIPerf produces **non-deterministic** results. Set `--random-seed <integer>` for reproducibility.

> [!WARNING]
> **Distributed System Constraints:** Even with `--random-seed`, **performance metrics and worker assignment are NOT reproducible** due to system non-determinism (network timing, async I/O, ZMQ load balancing).

**Reproducible (with `--random-seed`):**
- ✅ Dataset content (prompts, images, audio)
- ✅ Dataset sampling order (random/shuffle strategies)
- ✅ Request timing intervals (Poisson values)
- ✅ Model selection (random strategy)
- ✅ Session IDs (`session_000000`, `session_000001`, ...)

**NOT Reproducible (system-dependent):**
- ❌ Worker assignment / request execution order
- ❌ Performance metrics (TTFT, ITL, throughput)
- ❌ Server responses / absolute timestamps

**Testing:** Reproducibility is enforced by integration canary tests and CI/CD validation on every commit. See [Testing & Validation](#testing--validation).

## What Is Reproducible, What Is Not

**Key Principle:** Seeds control **WHAT you ask**, not **WHEN it completes** or **WHAT the server answers**.

### ✅ Reproducible with --random-seed

**Dataset:** Prompt text/tokens, image dimensions/formats, audio duration/formats, session IDs<br>
**Sampling:** Random selection, shuffle order, conversation selection<br>
**Timing Decisions:** Poisson interval values, cancellation decisions<br>

### ❌ NOT Reproducible

**Worker/Execution:** Which worker handles which request, request start/completion order, async I/O timing<br>
**Performance:** TTFT, ITL, latency, throughput<br>
**System:** Timestamps, process IDs, request IDs (ZMQ routing)<br>
**Server:** LLM output text, output token counts, errors/failures<br>

### Why This Architecture?

AIPerf achieves its high throughput through parallel workers, ZMQ load balancing, and async I/O. Full determinism would require single-worker synchronous execution, destroying performance.

### How It Works

**Phase 1 (Startup - PROFILE_CONFIGURE):**
- DatasetManager pre-generates complete dataset using derived RNGs and stores in memory
- TimingManager creates credit issuing strategy with RNG-based interval generator
- Workers set global seed (defensive measure) but don't derive/use RNGs

**Phase 2 (Runtime - PROFILE_START):**
- TimingManager generates intervals on-the-fly using RNG, sleeps, then drops credits
- Workers receive credits via ZMQ load balancing
- Workers request conversations from DatasetManager's pre-generated pool
- DatasetManager returns conversations (using sampler RNG or specific ID)
- Workers send API requests with pre-generated content
- **Result:** Same dataset and interval values, but actual timing/worker assignment vary per run

**Analogy:** Like a deterministic deck of cards (same 52 cards, same shuffle) dealt to players who play at different speeds. The deck is reproducible; card distribution to players varies based on who finishes hands first.

### Testing & Validation

Reproducibility is enforced by automated tests on every commit:

- **[test_random_generator_canary.py](../tests/integration/test_random_generator_canary.py)**: Compares payloads against reference snapshots to detect regressions
- **[test_deterministic_behavior.py](../tests/integration/test_deterministic_behavior.py)**: Verifies byte-for-byte identical outputs with same seed, different outputs with different seeds, tested with 5+ parallel workers

---

## User Guide

### Basic Usage

```bash
# Reproducible dataset
aiperf --random-seed 42 [options...]

# Non-reproducible (default)
aiperf [options...]
```

Same seed + same config = identical dataset content. Performance metrics always vary.

### Use Cases

**Debugging:** Reproduce exact prompts across runs to isolate prompt-related vs. network/timing issues
```bash
aiperf --random-seed 42 [...] --profile-export-file run1.json
aiperf --random-seed 42 [...] --profile-export-file run2.json
# Prompts identical; metrics may vary
```

**Performance Testing:** Compare metrics with same dataset
```bash
aiperf --random-seed 42 [...] --profile-export-file baseline.json
# After optimization...
aiperf --random-seed 42 [...] --profile-export-file optimized.json
# Use statistical analysis (median, p95, p99)
```

**Stress Testing:** Vary patterns by omitting seed
```bash
for i in {1..10}; do
  aiperf [...] --profile-export-file run_$i.json
done
```

## Developer Guide

### System Architecture

**Where RNGs Are Used:**
- **DatasetManager:** Pre-generates all dataset content at startup using derived RNGs
- **TimingManager:** Generates Poisson timing intervals and cancellation decisions
- **Workers:** Set global seed (defensive) but do NOT derive RNGs—they only execute API requests with pre-generated content

**Process Flow:**
1. `bootstrap.py` initializes RNG with `rng.init(seed)` in each process
   - Sets Python's `random.seed()` and NumPy's `np.random.seed()` globally (defensive measure)
   - Protects against third-party code inadvertently using global random state
2. DatasetManager creates generators (PromptGenerator, ImageGenerator, etc.) that derive RNGs in `__init__`
3. TimingManager creates interval generator that derives RNG in `__init__`
4. Workers initialize global seed but don't derive any RNGs (they only execute API requests)
5. All dataset content is generated before any requests are sent
6. Workers pull from pre-generated pool at runtime

### How to Use RNGs in Your Code

> [!IMPORTANT]
> Workers do NOT use RNGs. Only use RNGs in **DatasetManager** (content generation) or **TimingManager** (request timing) components.

```python
from aiperf.common import random_generator as rng

class MyGenerator:
    def __init__(self, config):
        # Derive once in __init__ with unique identifier
        self._rng = rng.derive("dataset.mycomponent.feature")

    def generate(self):
        # Use stored RNG instance
        return self._rng.choice([1, 2, 3, 4, 5])
```

**Rules:**
1. Derive in `__init__`, not in methods (or you'll get the same first value every call)
2. Store as instance variable
3. Use unique dotted identifier: `<module>.<component>.<aspect>`
4. Never use Python's `random` module (technically seeded, but fragile—any code using it affects your sequence)

### Hash-Based Seed Derivation

Uses SHA-256 to derive independent seeds: `SHA-256(root_seed:identifier)` → child seed

**Benefits:**
- Deterministic: Same identifier always gets same seed
- Independent: Changing one RNG doesn't affect others
- Fast: ~1-2 microseconds per derivation (happens once at init)

### Common Mistakes

**❌ Deriving in methods** → Returns same first value every call.<br>
**✅ Derive in `__init__`.**

**❌ Using Python's `random`** → Fragile (global state affected by any code).<br>
**✅ Use `rng.derive()`.**

**❌ Adding operations to existing RNG** → Shifts all subsequent values.<br>
**✅ Derive new RNG for new feature.**

## FAQ

**Q: Performance metrics still vary with same seed. Why?**<br>
A: Expected. Seeds control dataset content, not network timing or worker scheduling. See [What Is Reproducible](#what-is-reproducible-what-is-not).

**Q: Same seed across different configs?**<br>
A: Yes. Same seed + different config = different but reproducible results.

**Q: Multiple workers—how does this work?**<br>
A: Workers set global seed (defensive) but don't derive RNGs. DatasetManager pre-generates content, workers pull from this fixed pool. Validated with 5+ workers.

**Q: Are RNGs thread-safe?**<br>
A: No, but not an issue—each process uses RNGs in its own space. If adding multi-threaded RNG usage, derive per-thread.

**Q: Session IDs reproducible?**<br>
A: Yes. With seed: sequential (`session_000000`, `session_000001`). Without: UUIDs.

**Q: Performance impact?**<br>
A: None measurable. Network I/O dominates by 1000×.

## Reference

### All Component-Specific RNG Identifiers

**Dataset**
```python
# Prompts (3)
"dataset.prompt.length"        # Token count distribution
"dataset.prompt.corpus"        # Content position selection
"dataset.prompt.prefix"        # Prefix selection

# Images (3)
"dataset.image.dimensions"     # Width + height (coupled for aspect ratio)
"dataset.image.format"         # PNG/JPEG/etc. selection
"dataset.image.source"         # Source image selection

# Audio (3)
"dataset.audio.duration"       # Length distribution
"dataset.audio.format"         # Sample rate + bit depth
"dataset.audio.data"           # Audio sample generation

# Samplers (2)
"dataset.sampler.random"       # Random sampling strategy
"dataset.sampler.shuffle"      # Shuffle sampling strategy

# Loaders (2)
"dataset.loader.random_pool"   # Random pool loader
"dataset.loader.sharegpt"      # ShareGPT loader
```

**Timing**
```python
"timing.request.cancellation"      # Cancellation decisions (probabilistic)
"timing.request.poisson_interval"  # Exponential inter-arrival times (Poisson process)
```

**Composer**
```python
"composer.turn.model_selection"    # Model selection per turn
"composer.turn.max_tokens"         # max_tokens sampling
"composer.conversation.turn_count" # Number of turns per conversation
"composer.conversation.turn_delay" # Delay between turns
```

**Models**
```python
"models.sequence.distribution"     # ISL/OSL distribution sampling
```

### Module API

```python
from aiperf.common import random_generator as rng

# Initialize (called automatically in bootstrap.py)
rng.init(seed: int | None)
    # seed: Any integer for deterministic, None for random
    # Also sets global random.seed() and np.random.seed() defensively

# Derive component RNGs (call in __init__)
my_rng = rng.derive(identifier: str) -> RandomGenerator
    # Returns: Independent RNG with SHA-256 derived seed

# Reset (for testing only)
rng.reset()
```

See [random_generator.py](../src/aiperf/common/random_generator.py) for the RandomGenerator class and full API details.
