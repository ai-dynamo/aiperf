# Keep failed JobSet pods design

Date: 2026-03-28
Status: Proposed
Scope: Kubernetes benchmark debugging behavior for `AIPerfJob`
Compatibility: Default behavior remains unchanged unless the new debug flag is enabled.

## Summary

Failed worker and controller pod attempts are currently hard to debug because Kubernetes Job retry/backoff behavior deletes failed pod attempts before operators can inspect their logs and termination state. Increasing `ttlSecondsAfterFinished` does not solve this because TTL applies after Job or JobSet completion, while the Job controller may delete failed pod attempts during retry handling.

The design adds a per-job debug flag to preserve **all failed JobSet pod attempts** for that benchmark run. When enabled, the operator generates a JobSet that disables worker retries and omits Job and JobSet TTL cleanup so failed pods remain available for inspection.

## Problem statement

Current behavior:
- controller Jobs already use `backoffLimit=0`
- worker Jobs use `backoffLimit=3`
- worker Jobs may delete failed pod attempts during retry/backoff flow
- JobSet TTL and Job TTL only affect retention after completion

This means a failed benchmark can lose the exact failed pod that needs to be inspected, even if the overall JobSet object remains for several minutes.

## Design goals

1. Preserve **all failed JobSet pod attempts** for a single benchmark run when explicitly requested.
2. Keep current cleanup behavior unchanged for normal runs.
3. Make the behavior opt-in and visible in the CR spec.
4. Avoid global debug toggles that affect unrelated runs.
5. Keep the implementation minimal and local to the Kubernetes JobSet generation path.

## Proposed design

### 1. Add a per-job debug flag

Add a new top-level CR spec field:
- `spec.keepFailedPods: true | false`

Semantics:
- `false` or unset: current behavior
- `true`: preserve failed JobSet pod attempts for debugging

This field belongs at the top-level deployment/operator section of the CR because it controls Kubernetes JobSet behavior, not benchmark runtime logic.

### 2. Change JobSet generation only when enabled

When `spec.keepFailedPods` is `true`:
- controller Job `backoffLimit` remains `0`
- worker Job `backoffLimit` becomes `0`
- per-Job `ttlSecondsAfterFinished` is omitted
- JobSet-level `ttlSecondsAfterFinished` is omitted

Effect:
- failed worker/controller pod attempts are not deleted as part of retry handling because retries are disabled
- completed Job and JobSet objects are not auto-cleaned by TTL
- failed pod state, `kubectl describe pod`, current logs, and previous logs remain available for debugging

### 3. Default behavior remains unchanged

When `spec.keepFailedPods` is not set or is `false`:
- worker Jobs keep `backoffLimit=3`
- controller Jobs keep `backoffLimit=0`
- Job TTL behavior remains unchanged
- JobSet TTL behavior remains unchanged

This preserves current production behavior and avoids leaving failed debug artifacts around unless explicitly requested.

## Data flow

1. User submits `AIPerfJob` with optional `spec.keepFailedPods`
2. Operator validates and converts CR spec into deployment/jobset inputs
3. `JobSetSpec` receives the keep-failed-pods flag
4. JobSet manifest generation changes worker backoff and TTL emission only when the flag is enabled
5. Kubernetes retains failed pod attempts for that run

## Files to change

- `tools/generate_crd.py`
  - Add `keepFailedPods` to generated CRD schema
- `deploy/helm/aiperf-operator/templates/crd.yaml`
  - Regenerated output only
- `src/aiperf/operator/spec_converter.py`
  - Carry the new CR field into the deployment/jobset configuration path
- `src/aiperf/kubernetes/jobset.py`
  - Add the flag to `JobSetSpec`
  - Set worker backoff to `0` when enabled
  - Omit Job TTL and JobSet TTL when enabled
- tests covering CR/spec conversion and JobSet manifest generation

## Testing requirements

### Manifest generation

Add tests proving:
- default behavior still emits current TTLs and worker `backoffLimit=3`
- `keepFailedPods: true` yields:
  - worker `backoffLimit=0`
  - no worker Job TTL
  - no JobSet TTL

### Spec conversion / CR handling

Add tests proving:
- the CR spec accepts `keepFailedPods`
- the field flows into JobSet generation as expected

### CRD generation

Add checks proving:
- generated CRD schema includes `keepFailedPods`
- CRD output is regenerated and up to date

## Trade-offs

Pros:
- preserves exact failed pods for debugging
- explicit per-job control
- no impact on normal runs
- minimal implementation surface

Cons:
- debug runs can leave failed Jobs/pods behind until manually cleaned up
- disabling retries may reduce automatic recovery for transient worker failures during debug runs

## Final recommendation

Implement `spec.keepFailedPods: true` as an opt-in debug mode that disables worker retries and omits Job/JobSet TTL emission for that run. This is the smallest change that actually preserves failed pod attempts instead of only preserving higher-level Job objects after cleanup has already happened.
