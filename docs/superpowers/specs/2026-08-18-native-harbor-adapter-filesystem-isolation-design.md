<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Harbor adapter filesystem isolation design

## Problem

The current NativeGraph Docker adapter process is created with `docker exec` in
the task container. It has no model credentials and `no-network`, but it shares
the task's mutable worktree with declared artifacts. An adapter can therefore
write a declared artifact immediately after it reports a terminal transition.
The existing callback lifecycle collects and verifies artifacts before it reaps
the adapter, so the verifier can score that late mutation. Reaping before
collection is not safe either: the current adapter lease kills the *task*
container, making collection impossible.

The required invariant is stronger than process cleanup: an adapter must be
unable to mutate any artifact that the verifier will collect, regardless of
when it reports a protocol terminal event.

## Decision

Run the sealed NativeGraph environment adapter in a dedicated, task-owned
adapter container rather than through `docker exec` in the task container.
The task container remains the only container with the mutable task worktree.
The adapter container receives the exact sealed adapter argv over attached
stdio, an empty environment, no model secrets, `no-network`, and no task
worktree mount. It communicates exclusively through the existing strict
supervised-adapter protocol and artifact handles.

The adapter container is launched from the existing opaque task-minted adapter
start operation. The callback never receives Docker handles, a container name,
an argv, an environment map, or a mount option. Docker resolves and retains the
full immutable adapter-container ID after checking the task/run ownership
labels; every cancel, fence, and reap operation targets only that ID.

## Lifecycle

```text
resolved trial
  -> create/start task container (mutable task worktree, no-network)
  -> create/start labelled adapter container (no task worktree, no-network)
  -> attached strict protocol over adapter-container stdin/stdout
  -> reset / selected model decision / bounded step loop
  -> terminal transition
  -> stop and reap adapter container
  -> collect declared artifacts from still-running task container
  -> run independent verifier
  -> remove task container and remaining owned resources
```

On callback, protocol, adapter-start, or adapter-cleanup failure, Docker reaps
the adapter container and skips artifact collection and verification. A failed
task/container cleanup is reported with the primary failure; it cannot convert
an adapter failure into a scored result. The existing recovery ledger uses the
same labels and immutable ID to compensate uncertain adapter-container creates.

## Adapter inputs and capabilities

The first implementation uses the task image for the adapter container, but
does not mount the task worktree. The adapter's declared executable must be
available from the image itself. A task whose environment adapter depends on a
mutable `/work` checkout is rejected during sealed-start preflight rather than
silently receiving that mount. Read-only package inputs may be added only via a
future explicitly declared immutable snapshot capability; this slice does not
add one.

The adapter has no host Docker socket, task-container ID, model endpoint,
credential, or artifact filesystem path. It can obtain a frozen selected action
only through the existing one-shot artifact-read grant and can upload protocol
artifacts only through the bounded artifact store. It cannot create verifier
inputs by writing the task worktree.

## Docker representation

The adapter container receives the existing exact run labels plus an explicit
adapter role label. Its creation uses a dedicated Docker request builder rather
than `ContainerWorkspace`, so absence of a workspace mount is structural rather
than convention. It runs attached (`docker start --attach --interactive`, or an
equivalent created-container attach API) so Rust retains bounded stdin, stdout,
and stderr supervision. It is started with a finite startup deadline and the
existing task-minted adapter deadlines.

The task container is not killed when the adapter is canceled. The new adapter
lease owns only the adapter-container ID. This replaces the old `docker exec`
lease, whose `docker kill` targeted the task container.

## Compatibility and scope

This changes only sealed NativeGraph rollout adapters. Legacy non-rollout
adapter starts retain their current behavior. Compose, external-driver,
cross-host cellular, and a general adapter-image registry are out of scope.
No raw model responses, raw adapter traffic, or mutable filesystem paths enter
rollout evidence.

## Tests

The implementation must add a real Docker end-to-end regression where an
environment adapter attempts to overwrite `/work/result.txt` after sending a
terminal transition. The adapter container must have no writable task worktree;
the verifier must observe the original task result and score the action-derived
outcome. The test also checks selected-model request count, exact adapter labels,
empty adapter environment, `no-network`, adapter reap before collection, and
task-container removal after verification.

Additional focused tests cover: missing image-resident adapter executable
preflight refusal; callback/protocol failure reaping only the adapter container;
wrong/missing ownership labels refusing before a kill; no verifier after adapter
failure; and recovery after an uncertain adapter-container create.
