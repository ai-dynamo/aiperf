# Origin #45 finding: high-resolution request-rate pacing

Upstream commit `21f8ad7b3e621285a1682b336df16607e7d3bb9f` fixes
high-rate under-delivery in the Python request-rate loop. It combines two
changes: a reusable absolute-deadline pacer (Linux `timerfd`, with a sleep-thread
fallback) and a bounded 10 ms catch-up window before a late schedule re-anchors.
It also adds two Python timing environment settings and focused unit coverage.

## Source/native comparison

| Upstream behavior | Native state at base `f423b618da` | Disposition |
|---|---|---|
| Bypass the Python event-loop timer wheel | `RealClock::sleep` already uses a Linux `CLOCK_MONOTONIC` `timerfd` through Tokio `AsyncFd`, with owned-descriptor cleanup and a Tokio fallback on syscall/reactor failure. | Already covered; do not add a second pacer, thread, or per-tick synchronization layer. |
| Reuse one pacing descriptor | Native opens one owned timerfd per positive sleep. The descriptor is closed by RAII on completion, cancellation, or error, but reuse would require a separate single-waiter clock outside the shared concurrent `Clock` contract. | Not imported in this tracker: resolution is already native, and a reusable timer belongs in a separately benchmarked clock redesign rather than the rate-policy fix. |
| Preserve small late slots and re-anchor only after a bounded backlog | The local/sharded loop at `rust/runtime/src/request_rate.rs` re-anchors after every positive oversleep, forfeiting sub-millisecond slots. | Applicable. Preserve targets up to the configured lag window and re-anchor only beyond it. |
| Bound catch-up bursts | The shared global rate gate intentionally retains every dense slot for exact aggregate/corpus ordering; the local loop has no bounded window. | Apply the new window to the local/sharded renewal loop. Keep the global gate's stronger dense-slot contract unchanged; silently dropping or shifting shared claims would break global exactness. |
| `AIPERF_TIMING_MAX_CATCHUP_SECONDS`, default `0.01`, range `0..=10` | Native has no matching capture or validation. | Applicable. Capture once when the workload is constructed, validate finite range, convert once to integer nanoseconds, and keep environment reads off the issuance hot path. |
| `AIPERF_TIMING_HIGH_RES_TIMER=false` diagnostic fallback | Native has one injected `Clock` seam, not Python's alternate event-loop-vs-pacer stacks. | Not applicable. Adding a Tokio-timer bypass around `Clock` would violate clock injection and make `SimClock` behavior diverge. |

Disposition: **applicable, partially already covered**. The native port is the
bounded local/sharded late-slot policy plus exact high-rate evidence over the
existing `RealClock`; it is not a line-for-line port of Python-specific timer
objects.

## Upstream test mapping

- Timerfd wake-not-before and non-positive sleep behavior map to existing
  `clock::real_clock` tests; owned `OwnedFd` plus `AsyncFd` provide cancellation
  and error cleanup without an explicit `close()` API.
- Python pacer selection and thread cancellation are not applicable because
  native has neither a pacer factory nor a sleep thread.
- Rate-update schedule ordering remains covered by the native interval/ramp
  actuator tests; the current tick is immutable and the next interval is drawn
  before dispatch.
- New deterministic Rust tests must prove literal below-window preservation,
  beyond-window re-anchoring, zero-window no-burst behavior, saturated
  arithmetic, and environment parsing.
- A Rust integration characterization must use the real `RealClock`, issue an
  exact bounded request count at a sub-millisecond constant interval, and assert
  count plus elapsed-rate bounds. It must not assert an unrealistically tight
  per-wakeup latency on a shared CI host.

The upstream commit adds no Python integration or end-to-end test. The real
clock integration is therefore the applicable native product-level evidence.

## Ancestry and scope

Merge `86a93aaec1` is a target-only two-parent merge with the exact upstream
commit as its second parent. Its first-parent diff is exactly the upstream
seven-file Python/docs delta (856 insertions, 82 deletions); TraceLab #44 and
other cumulative upstream changes are absent. No cherry-pick was used.

Native implementation and final verification evidence are pending the TDD,
task review, whole-branch review, and Graham gates recorded in the linked plan.
