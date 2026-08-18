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
The task container remains the verifier's mutable task worktree. The adapter
container has a separate private working copy, receives the exact sealed
adapter argv over attached stdio, an empty environment, no model secrets, and
`no-network`.

The adapter never mutates the verifier workspace directly. Before each
authoritative transition it uploads one bounded typed workspace-patch artifact.
Rust validates that artifact, applies it to the verifier workspace only after
accepting the matching transition, and records its frozen digest in the
descriptor-only rollout receipt. Once a terminal transition is accepted, Rust
accepts no more patches. Later adapter writes affect only its private working
copy and cannot change verification.

The adapter container is launched from the existing opaque task-minted adapter
start operation. The callback never receives Docker handles, a container name,
an argv, an environment map, or a mount option. Docker resolves and retains the
full immutable adapter-container ID after checking the task/run ownership
labels; every cancel, fence, and reap operation targets only that ID.

## Lifecycle

```text
resolved trial
  -> create/start task container (verifier worktree, no-network)
  -> create/start labelled adapter container (private worktree, no-network)
  -> attached strict protocol over adapter-container stdin/stdout
  -> reset / selected model decision / bounded step loop
     -> adapter uploads bounded patch -> Rust validates and commits it
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

## Workspace patch protocol

A sealed rollout environment contract declares explicit mutable paths and
limits: a finite positive maximum patch count, maximum total patch bytes, and
maximum bytes per patch. Each patch is a strict archive with relative normalized
paths only. Rust rejects absolute paths, `..`, symlinks, device nodes, hard
links, special modes, duplicate paths, paths outside the declared mutable set,
and every size/count/total-limit violation before materializing a file.

The adapter uploads a patch through the existing bounded artifact protocol and
references it in the same operation as its `Transition`. Rust admits it only
when it matches the current session, action, operation, and observation. It
materializes into a fresh host-controlled staging directory, validates the full
archive, then applies it atomically to the task workspace. A failed patch or
apply aborts the episode before collection and verification. The frozen rollout
receipt retains only the patch digest and descriptor identities, never archive
paths or payloads.

## Adapter inputs and capabilities

The first implementation uses the task image for the adapter container and an
owned private workspace mount populated from immutable imported source. It
never mounts the verifier workspace. The sealed adapter fields have separate
roles: `executable` remains the package-relative source-provenance file, while
Docker rollout `argv[0]` must be an absolute path available in the built image.
A task whose Docker rollout command is relative or under mutable `/work` is
rejected during sealed-start preflight rather than silently receiving a mount.

The adapter has no host Docker socket, task-container ID, model endpoint,
credential, verifier-workspace path, or artifact filesystem path. It can obtain
a frozen selected action only through the existing one-shot artifact-read grant,
and can upload observation/info and typed workspace-patch artifacts through the
bounded artifact store. It cannot create verifier inputs by writing the task
worktree.

## Docker representation

The adapter container receives the existing exact run labels plus an explicit
adapter role label. Its creation uses a dedicated Docker request builder rather
than `ContainerWorkspace`, so a verifier-workspace mount is structurally
impossible. The builder may mount only the separately created private adapter
workspace. It runs attached (`docker start --attach --interactive`, or an
equivalent created-container attach API) so Rust retains bounded stdin, stdout,
and stderr supervision. It is started with a finite startup deadline and the
existing task-minted adapter deadlines.

The task container is not killed when the adapter is canceled. The new adapter
lease owns only the adapter-container ID. This replaces the old `docker exec`
lease, whose `docker kill` targeted the task container.

## Compatibility and scope

This changes only sealed NativeGraph rollout adapters. Legacy non-rollout
adapter starts retain their current behavior. Compose, external-driver,
cross-host cellular, reusable private workspaces, and a general adapter-image
registry are out of scope. No raw model responses, raw adapter traffic, archive
payloads, or mutable filesystem paths enter rollout evidence.

## Tests

The implementation must add a real Docker end-to-end regression where an
environment adapter attempts to overwrite its private `/work/result.txt` after
sending a terminal transition. The adapter must first upload the action-derived
patch that produces `south`; the verifier must observe committed `south` rather
than the late private `north` mutation. The test also checks selected-model
request count, exact adapter labels, empty adapter environment, `no-network`,
adapter reap before collection, and task-container removal after verification.

Additional focused tests cover malformed, oversized, unsafe-path, symlink,
wrong-operation, replayed, and post-terminal patches; missing image-resident
adapter executable preflight refusal; callback/protocol failure reaping only the
adapter container; wrong/missing ownership labels refusing before a kill; no
verifier after adapter failure; and recovery after an uncertain adapter-container
create.
