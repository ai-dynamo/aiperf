<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Prefix Synthesis API Reference

Complete API documentation for the prefix synthesis module.

## Module: `aiperf.dataset.synthesis`

### Classes

#### `RollingHasher`

Converts sequences of text blocks into globally unique hash IDs using rolling hash.

**Constructor:**
```python
RollingHasher(block_size: int = 512) -> None
```

**Parameters:**
- `block_size` (int): Number of tokens per block for hashing (default: 512)

**Methods:**

`hash_blocks(blocks: Sequence[str]) -> list[int]`
- Convert a sequence of text blocks to hash IDs
- **Parameters:**
  - `blocks`: Sequence of text strings representing blocks
- **Returns:** List of unique hash IDs

`hash_token_blocks(blocks: Sequence[Sequence[int]]) -> list[int]`
- Convert a sequence of token blocks to hash IDs
- **Parameters:**
  - `blocks`: Sequence of token blocks (each block is a sequence of token IDs)
- **Returns:** List of unique hash IDs

`reset() -> None`
- Reset the hasher state for hashing new sequences
- Note: Maintains global uniqueness across sequences (keeps `_hash_to_id` and `_id_counter`)

`get_stats() -> dict[str, int]`
- Get statistics about the hasher
- **Returns:** Dictionary with 'total_hashes' and 'max_id'

**Example:**
```python
from aiperf.dataset.synthesis import RollingHasher

hasher = RollingHasher(block_size=512)
hash_ids = hasher.hash_blocks(["hello", "world", "test"])
# Result: [0, 1, 2]

stats = hasher.get_stats()
# {'total_hashes': 3, 'max_id': 2}
```

---

#### `RadixTree`

Compact representation of prefix patterns using a radix tree data structure.

**Constructor:**
```python
RadixTree() -> None
```

**Methods:**

`add_path(path: list[int]) -> RadixNode`
- Add a path to the tree from root
- **Parameters:**
  - `path`: List of edge labels (hash IDs or token counts)
- **Returns:** The leaf RadixNode at the end of the path
- **Side effects:** Increments visit_count for all nodes in the path

`get_node(node_id: int) -> RadixNode | None`
- Get node by ID
- **Returns:** RadixNode or None if not found

`get_all_nodes() -> list[RadixNode]`
- Get all nodes in the tree
- **Returns:** List of all RadixNode instances

`get_stats() -> RadixTreeStats`
- Get statistics about tree structure
- **Returns:** `RadixTreeStats` object with 'num_nodes', 'num_leaves', 'total_visits', 'max_depth'

**Properties:**

`root: RadixNode`
- The root node of the tree (read-only)

**Example:**
```python
from aiperf.dataset.synthesis import RadixTree

tree = RadixTree()

# Add paths
tree.add_path([1, 2, 3])
tree.add_path([1, 2, 4])
tree.add_path([1, 5, 6])

# Get statistics
stats = tree.get_stats()
# {
#   'num_nodes': 7,
#   'num_leaves': 3,
#   'total_visits': 3,
#   'max_depth': 3
# }
```

---

#### `RadixNode`

A node in the radix tree representing a prefix path.

**Constructor:**
```python
RadixNode(
    node_id: int,
    label: int | None = None,
    visit_count: int = 0,
    children: dict[int, RadixNode] | None = None,
    parent: RadixNode | None = None
) -> None
```

**Attributes:**
- `node_id`: Unique node identifier
- `label`: Edge label (token count)
- `visit_count`: Number of times this node is visited
- `children`: Dictionary of child nodes by edge label
- `parent`: Parent node reference

**Methods:**

`add_child(label: int, child: RadixNode) -> None`
- Add a child node with the given edge label

`get_child(label: int) -> RadixNode | None`
- Get child with given label

`is_leaf() -> bool`
- Check if this node is a leaf (no children)

---

#### `EmpiricalSampler`

Samples values from an empirical distribution learned from data.

**Constructor:**
```python
EmpiricalSampler(data: list[int] | list[float]) -> None
```

**Parameters:**
- `data`: List of observed values to learn distribution from

**Methods:**

`sample() -> int | float`
- Draw a single sample from the learned distribution
- **Returns:** A sampled value

`sample_batch(size: int) -> list[int | float]`
- Draw multiple samples from the learned distribution
- **Parameters:**
  - `size`: Number of samples to draw
- **Returns:** List of sampled values

`get_stats() -> EmpiricalSamplerStats`
- Get statistics about the learned distribution
- **Returns:** `EmpiricalSamplerStats` object with 'min', 'max', 'mean', 'median', 'num_unique'

**Example:**
```python
from aiperf.dataset.synthesis import EmpiricalSampler, EmpiricalSamplerStats

# Create sampler from observed data
data = [100, 200, 150, 300, 250, 100, 200]
sampler = EmpiricalSampler(data)

# Sample from distribution
sample = sampler.sample()  # Returns a value like 100, 150, 200, 250, or 300

# Get distribution statistics
stats = sampler.get_stats()
# EmpiricalSamplerStats(min=100.0, max=300.0, mean=185.7, median=200.0, num_unique=5)
```

---

#### `PrefixAnalyzer`

Analyzes traces to extract ISL/OSL statistics and prefix patterns.

**Constructor:**
```python
PrefixAnalyzer(block_size: int = 512) -> None
```

**Parameters:**
- `block_size` (int): Number of tokens per block for analysis (default: 512)

**Methods:**

`analyze_file(trace_file: Path | str) -> AnalysisStats`
- Analyze a mooncake trace file
- **Parameters:**
  - `trace_file`: Path to JSONL trace file
- **Returns:** AnalysisStats object

`analyze_traces(traces: list[dict]) -> AnalysisStats`
- Analyze a list of trace dictionaries
- **Parameters:**
  - `traces`: List of trace dictionaries
- **Returns:** AnalysisStats object

**Example:**
```python
from aiperf.dataset.synthesis import PrefixAnalyzer
from pathlib import Path

analyzer = PrefixAnalyzer(block_size=512)

# Analyze from file
stats = analyzer.analyze_file("traces/production.jsonl")

print(f"Total requests: {stats.total_requests}")
print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")
print(f"Average ISL: {stats.avg_isl:.1f}")
```

---

#### `Synthesizer`

Generates synthetic traces preserving prefix-sharing patterns.

**Constructor:**
```python
Synthesizer(params: SynthesisParams | None = None) -> None
```

**Parameters:**
- `params` (SynthesisParams): Generation configuration (optional, uses defaults if None)

**Methods:**

`synthesize_from_file(trace_file: Path | str) -> list[dict]`
- Synthesize traces from an input trace file
- **Parameters:**
  - `trace_file`: Path to input JSONL trace file
- **Returns:** List of synthetic trace dictionaries

`synthesize_traces(traces: list[dict]) -> list[dict]`
- Synthesize traces from a list of trace dictionaries
- **Parameters:**
  - `traces`: List of input trace dictionaries
- **Returns:** List of synthetic trace dictionaries

`synthesize_grouped_traces(data: dict[str, list[dict]]) -> dict[str, list[dict]]`
- Synthesize traces while preserving session grouping
- **Parameters:**
  - `data`: Dictionary mapping session_id to list of trace dicts
- **Returns:** Dictionary mapping session_id to list of synthesized trace dicts

`get_stats() -> dict[str, Any]`
- Get synthesizer statistics
- **Returns:** Dictionary with 'tree_nodes', 'tree_depth', 'params'

**Example:**
```python
from aiperf.dataset.synthesis import Synthesizer
from aiperf.dataset.synthesis.models import SynthesisParams

# Create synthesizer with custom parameters
params = SynthesisParams(
    speedup_ratio=2.0,
    prefix_len_multiplier=1.5,
    max_isl=4096
)
synthesizer = Synthesizer(params=params)

# Synthesize from file
synthetic_traces = synthesizer.synthesize_from_file("input.jsonl")

# Synthesize from list
synthetic_traces = synthesizer.synthesize_traces([
    {"input_length": 512, "output_length": 64, "hash_ids": [1, 2, 3]},
    {"input_length": 768, "output_length": 128, "hash_ids": [1, 2, 4]},
])

# Get statistics
stats = synthesizer.get_stats()
print(f"Tree nodes: {stats['tree_nodes']}")
```

---

### Data Models

#### `AnalysisStats`

Statistics extracted from trace analysis.

**Fields:**
- `total_requests: int` - Total number of requests in trace
- `unique_prefixes: int` - Number of unique prefix patterns
- `cache_hit_rate: float` - Theoretical cache hit rate (0.0 to 1.0)
- `min_isl: int` - Minimum input sequence length
- `max_isl: int` - Maximum input sequence length
- `avg_isl: float` - Average input sequence length
- `min_osl: int` - Minimum output sequence length
- `max_osl: int` - Maximum output sequence length
- `avg_osl: float` - Average output sequence length
- `prefix_reuse_ratio: float` - Ratio of reused prefixes (0.0 to 1.0)
- `isl_stats: MetricStats | None` - Full statistics for input sequence length
- `osl_stats: MetricStats | None` - Full statistics for output sequence length
- `context_length_stats: MetricStats | None` - Full statistics for context (shared prefix) length
- `unique_prompt_length_stats: MetricStats | None` - Full statistics for unique prompt length
- `hit_rate_stats: MetricStats | None` - Full statistics for per-request cache hit rates

#### `MetricStats`

Statistics for a single metric with percentiles.

**Fields:**
- `mean: float` - Mean value
- `std_dev: float` - Standard deviation
- `min: float` - Minimum value
- `p25: float` - 25th percentile
- `median: float` - Median (50th percentile)
- `p75: float` - 75th percentile
- `max: float` - Maximum value

#### `SynthesisParams`

Parameters for synthetic trace generation.

**Fields:**
- `speedup_ratio: float = 1.0` - Timestamp scaling multiplier (ge 0.0)
- `prefix_len_multiplier: float = 1.0` - Core prefix length multiplier (ge 0.0)
- `prefix_root_multiplier: int = 1` - Number of independent trees to distribute traces across (ge 1)
- `prompt_len_multiplier: float = 1.0` - Leaf prompt length multiplier (ge 0.0)
- `max_isl: int | None = None` - Maximum input sequence length filter
- `block_size: int = 512` - KV cache page size (ge 1)

**Class Methods:**
- `from_synthesis_config(config: SynthesisConfig, block_size: int = 512) -> SynthesisParams` - Create from SynthesisConfig

#### `RadixTreeStats`

Statistics about radix tree structure.

**Fields:**
- `num_nodes: int` - Total number of nodes in tree
- `num_leaves: int` - Number of leaf nodes (nodes with no children)
- `total_visits: int` - Number of paths added to tree
- `max_depth: int` - Maximum depth from root to leaf

#### `EmpiricalSamplerStats`

Statistics about learned empirical distribution.

**Fields:**
- `min: float` - Minimum value in original data
- `max: float` - Maximum value in original data
- `mean: float` - Mean of original data
- `median: float` - Median of original data
- `num_unique: int` - Number of unique values in distribution

---

### Utility Functions

#### `aiperf.dataset.synthesis.graph_utils`

Graph manipulation utilities for radix tree operations.

`validate_tree(tree: RadixTree) -> bool`
- Validate tree structure consistency
- Checks parent-child relationships and reachability

`remove_leaves(tree: RadixTree, visit_threshold: int = 1) -> None`
- Remove leaf nodes visited N times or less
- **Parameters:**
  - `tree`: RadixTree to prune
  - `visit_threshold`: Minimum visit count to keep

`merge_unary_chains(tree: RadixTree) -> None`
- Merge unary chains (nodes with single children) into compressed edges
- Modifies tree in-place

`compute_transition_cdfs(tree: RadixTree) -> dict[int, np.ndarray]`
- Compute cumulative distribution functions for outgoing transitions
- **Returns:** Dictionary mapping node IDs to CDF arrays

`get_tree_stats(tree: RadixTree) -> dict[str, Any]`
- Get comprehensive statistics about tree structure
- **Returns:** Dictionary with tree metrics (includes RadixTreeStats fields plus 'internal_nodes' and 'branching_factor')

#### `aiperf.dataset.synthesis.rolling_hasher`

Utility functions for converting between texts and hash IDs.

`texts_to_hashes(tokenizer: Tokenizer, texts: list[str], block_size: int = 512) -> list[list[int]]`
- Convert a list of texts to hash ID sequences
- Tokenizes texts, splits into blocks, and generates consecutive hash IDs
- **Parameters:**
  - `tokenizer`: Tokenizer for encoding texts
  - `texts`: List of input text strings
  - `block_size`: Number of tokens per block
- **Returns:** List of hash ID sequences, one per input text

`hashes_to_texts(prompt_generator: PromptGenerator, hash_ids_list: list[list[int]], input_lengths: list[int], block_size: int = 512) -> list[str]`
- Convert hash ID sequences back to text strings
- Uses the PromptGenerator's cache to ensure the same hash ID always produces the same token block
- **Parameters:**
  - `prompt_generator`: PromptGenerator instance for generating text from hash_ids
  - `hash_ids_list`: List of hash ID sequences
  - `input_lengths`: Target input lengths (in tokens) for each sequence
  - `block_size`: Number of tokens per block
- **Returns:** List of text strings, one per hash ID sequence
- **Raises:** `ValueError` if `len(hash_ids) * block_size < input_length` for any sequence

---

## CLI Commands

### `aiperf analyze-trace`

Analyze a mooncake trace file for ISL/OSL distributions and cache hit rates.

**Usage:**
```bash
aiperf analyze-trace INPUT_FILE [OPTIONS]
```

**Arguments:**
- `INPUT_FILE`: Path to input mooncake trace JSONL file

**Options:**
- `--block-size INT` (default: 512) - KV cache block size
- `--output-file PATH` - Optional output path for analysis report (JSON)

**Example:**
```bash
aiperf analyze-trace traces/prod.jsonl --output-file analysis.json
```

---

## Configuration

Synthesis parameters can be configured via CLI or programmatically.

**Via CLI Options (with `aiperf profile`):**

Synthesis is applied automatically when running `aiperf profile` with mooncake traces and synthesis parameters:

```bash
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-speedup-ratio 2.0 \
    --synthesis-prefix-len-multiplier 1.5 \
    --synthesis-max-isl 4096 \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat
```

**Via SynthesisConfig:**
```python
from aiperf.common.config import SynthesisConfig

config = SynthesisConfig(
    speedup_ratio=2.0,
    prefix_len_multiplier=1.5,
    max_isl=4096,
)

# Check if synthesis would be triggered
if config.should_synthesize():
    print("Synthesis will be applied")
```

**Direct Instantiation:**
```python
from aiperf.dataset.synthesis import Synthesizer
from aiperf.dataset.synthesis.models import SynthesisParams

params = SynthesisParams(
    speedup_ratio=2.0,
    prefix_len_multiplier=1.5,
    max_isl=4096,
)

synthesizer = Synthesizer(params=params)
synthetic_traces = synthesizer.synthesize_from_file("input.jsonl")
```

---

## Integration with AIPerf Benchmark

Synthesis is applied automatically during benchmark when using mooncake traces with synthesis parameters:

```bash
# Run benchmark with synthesis applied in-memory
aiperf profile \
    --input-file production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-speedup-ratio 2.0 \
    --synthesis-prefix-len-multiplier 1.5 \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat
```

Synthesis is triggered automatically when any `--synthesis-*` parameter differs from its default value.

---

## Performance Considerations

- **Memory**: Radix tree construction can be memory-intensive for large traces (>100k requests)
- **Time**: Analysis is O(n) where n is number of requests
- **Synthesis**: O(n) to generate synthetic traces from analyzed data
- **Optimization**: Use `remove_leaves()` to prune infrequent paths for memory savings

---

## Error Handling

All components use standard Python exceptions:

```python
try:
    stats = analyzer.analyze_file("nonexistent.jsonl")
except FileNotFoundError:
    print("Trace file not found")
except ValueError as e:
    print(f"Invalid trace format: {e}")
```

---

## See Also

- [Prefix Synthesis Tutorial](../tutorials/prefix-synthesis.md)
- [Trace Replay](../benchmark_modes/trace_replay.md)
