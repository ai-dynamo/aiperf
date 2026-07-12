# Planned GitHub review for ai-dynamo/aiperf#1036

Status: draft only; do not post without explicit user approval.

## Top-level review body

Requesting changes because the current recorder can silently corrupt its final Parquet data. Fix order: (1) isolate histogram label sets and make Arrow checkpoints non-overlapping with numbered Parquet files, (2) retain Prometheus values as Float64 and reject non-2xx responses, then (3) replace the comma-split label parser with a standards-compliant parser. I reproduced all five cases with the PR's compiled `tachometer-scraper` binary.

The overall product split is sensible: a thin `aiperf watch` wrapper over an independently useful local/S3 Prometheus archive. The Rust workspace is organized clearly, the `MetricFilter` seam is useful, Parquet publication is atomic, shutdown compacts retained data, and all 43 Rust tests pass. These data-integrity issues need coverage before the archive is safe to rely on.

## Inline 1 — histogram label-set contamination

- Path: `rust/tachometer-scraper/src/parse.rs`
- Line/position: 548
- Side: RIGHT

**[P1] Keep each histogram label set in a separate bucket group**

`histogram_stats` is keyed by family plus non-`le` labels, but this map groups rows only by the metric-name prefix. With `route="a"` and `route="b"`, the combined bucket sort makes one route's lower bound depend on the other, then lines 568-589 select the first route's sum/count for every bucket. The compiled PR binary stored route B's expected `sum=70,count=10` as `sum=1.5,count=3` and gave its first bucket a lower bound of 1 instead of 0. Please group by the same `(family, labels_without_le)` identity used by the stats map and compute bounds independently per series.

## Inline 2 — stale checkpoint duplication

- Path: `rust/tachometer-writer/src/writer.rs`
- Line/position: 279
- Side: RIGHT

**[P1] Do not leave a checkpoint for rows already committed to Parquet**

The periodic task can snapshot the current buffer to `current.arrow`; a later threshold flush writes those same rows to `out-N.parquet` and clears the buffer. When shutdown reaches this empty-buffer branch, the stale Arrow file remains, and final compaction concatenates both sources. In a two-scrape reproduction, `out-1.parquet` had two rows, `current.arrow` retained the first, and `final.parquet` had three rows with the first timestamp duplicated. Please invalidate the checkpoint atomically when its buffer generation is committed, or track committed offsets so compaction excludes superseded checkpoint rows.

## Inline 3 — Float32 precision loss

- Path: `rust/tachometer-writer/src/writer.rs`
- Line/position: 18
- Side: RIGHT

**[P1] Preserve Prometheus numeric values as Float64**

The parser starts with `f64`, but `Row` and the Arrow schema downcast every value, histogram bound, sum, and count to Float32. Float32 stops representing consecutive integers above 16,777,216, which is well below common request/token/energy counters. The compiled PR binary stored a source counter of `100000001` as `100000000.0`, so downstream deltas can disappear. Please use `f64`/Arrow Float64 end to end.

## Inline 4 — quoted-label parsing

- Path: `rust/tachometer-scraper/src/parse.rs`
- Line/position: 417
- Side: RIGHT

**[P2] Parse quoted Prometheus labels without splitting embedded commas**

Prometheus label values may contain escaped commas, quotes, backslashes, and newlines. Splitting the raw label block on every comma changes valid series identities: the compiled PR binary turned `message="left,right"` into `message="left"` and silently discarded the remainder. Please use a Prometheus/OpenMetrics parser or a quote/escape-aware state machine, and escape values again when formatting labels.

## Inline 5 — HTTP status validation

- Path: `rust/tachometer-scraper/src/lib.rs`
- Line/position: 28
- Side: RIGHT

**[P1] Reject non-success HTTP responses before parsing the body**

`reqwest::get` does not treat 4xx/5xx as errors, so this path archives a failure body whenever it happens to look like Prometheus text; other error pages are silently reduced to empty scrapes by the parser. A localhost server returning HTTP 500 with `error_metric 42` produced a normal Parquet row with value 42 and no scrape error. Please call `error_for_status` (or check the status explicitly) and use a reusable client with connection/request timeouts.
