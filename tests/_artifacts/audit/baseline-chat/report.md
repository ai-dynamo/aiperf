# Audit Report: `baseline-chat` — FAIL

## Exact

_(no findings)_

## Tolerance

| field | expected | actual | reason |
| --- | --- | --- | --- |
| `request_throughput.avg` | `1854.929714829192` | `2135.059386510034` | relative diff 13.1% exceeds band 10.0% for stat 'avg' |
| `request_latency.avg` | `1.516322453125` | `0.6898681249999999` | relative diff 54.5% exceeds band 10.0% for stat 'avg' |
| `request_latency.p1` | `0.38290164` | `0.32744534000000003` | relative diff 14.5% exceeds band 10.0% for stat 'p1' |
| `request_latency.p10` | `0.4393838` | `0.39265849999999997` | relative diff 10.6% exceeds band 10.0% for stat 'p10' |
| `request_latency.p25` | `0.6281439999999999` | `0.46311674999999997` | relative diff 26.3% exceeds band 10.0% for stat 'p25' |
| `request_latency.p50` | `0.6734304999999999` | `0.5692265` | relative diff 15.5% exceeds band 10.0% for stat 'p50' |
| `request_latency.p75` | `0.83794675` | `0.67069475` | relative diff 20.0% exceeds band 10.0% for stat 'p75' |
| `request_latency.p95` | `10.865638499999939` | `1.6231166499999978` | relative diff 85.1% exceeds band 25.0% for stat 'p95' |
| `request_latency.p99` | `15.14387014` | `2.612682369999998` | relative diff 82.7% exceeds band 25.0% for stat 'p99' |
| `request_latency.max` | `15.148168` | `3.092806` | relative diff 79.6% exceeds band 25.0% for stat 'max' |
| `request_latency.std` | `3.2372036172079475` | `0.4745344630603931` | relative diff 85.3% exceeds band 50.0% for stat 'std' |
| `output_token_throughput.avg` | `1022443.0554692096` | `1162006.071108086` | relative diff 12.0% exceeds band 10.0% for stat 'avg' |
| `benchmark_duration.avg` | `0.034502655` | `0.029975747` | relative diff 13.1% exceeds band 10.0% for stat 'avg' |
| `e2e_output_token_throughput.avg` | `818779.8208254927` | `974592.0170759889` | relative diff 16.0% exceeds band 10.0% for stat 'avg' |
| `e2e_output_token_throughput.p1` | `29621.052783192183` | `192317.2137447797` | relative diff 84.6% exceeds band 10.0% for stat 'p1' |
| `e2e_output_token_throughput.p5` | `89500.25004117207` | `274577.72211093013` | relative diff 67.4% exceeds band 10.0% for stat 'p5' |
| `e2e_output_token_throughput.p25` | `671358.0832890533` | `774452.0922098206` | relative diff 13.3% exceeds band 10.0% for stat 'p25' |
| `e2e_output_token_throughput.p50` | `836612.9927126995` | `1035687.2021664795` | relative diff 19.2% exceeds band 10.0% for stat 'p50' |
| `e2e_output_token_throughput.p75` | `940220.0069585284` | `1174446.3146348496` | relative diff 19.9% exceeds band 10.0% for stat 'p75' |
| `e2e_output_token_throughput.min` | `29587.858305331636` | `161342.15983802412` | relative diff 81.7% exceeds band 25.0% for stat 'min' |
| `usage_completion_tokens_diff_pct.p1` | `0.1665640599001664` | `0.1881683842115497` | relative diff 11.5% exceeds band 10.0% for stat 'p1' |
| `usage_completion_tokens_diff_pct.p5` | `0.6148653359329361` | `0.4746077215102256` | relative diff 22.8% exceeds band 10.0% for stat 'p5' |
| `total_token_throughput.avg` | `2042654.398625265` | `2336288.733688605` | relative diff 12.6% exceeds band 10.0% for stat 'avg' |

## Structural

_(no findings)_
