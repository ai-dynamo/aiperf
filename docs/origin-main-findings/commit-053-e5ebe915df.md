# Origin #53 findings: case-insensitive enum inequality

Upstream `e5ebe915df38958b2a7d36eb40bb1d407e991ead` fixes a Python data-model
bug in `CaseInsensitiveStrEnum` and `ExtensibleStrEnum`.  Both classes override
`__eq__` to fold case and `-`/`_`, but inherit `str.__ne__`; before the fix,
`member == "FOO-BAR"` and `member != "FOO-BAR"` were both true.

## Ancestry is not behavior

`git merge-base --is-ancestor e5ebe915df HEAD` succeeds at campaign base
`ecf575c3cf31f69c23ec63715300ecd29f5d0792`, but the resulting tree had
regressed the complete upstream delta.  Comparing the current tree with the
upstream commit found the two `__ne__` methods and their test coverage absent.
The direct product reproduction before restoration printed `True True` for
both equality operators against `"foo-bar"` for each enum family.

This is therefore an applicable shared-product restoration, not an
ancestry-only or Python-only closure.  A new restoration commit is required:
Git cannot create another actual merge of `e5ebe915df` because it is already an
ancestor, while the tree-level semantic conflict had discarded its changes.

## Native comparison

Rust has no `str`-subclass/MRO enum mechanism.  The analogous
case/hyphen-insensitive identifiers are canonicalized at construction:

- `rust/runtime/src/extensions/registry_id.rs` normalizes into `RegistryId`
  and derives `Eq` and `PartialEq`.
- `rust/runtime/src/engine/protocol_v2.rs` normalizes into `ComponentId` and
  derives `PartialEq` and `Eq`.

Rust derives `!=` as the logical negation of `==`; no independently inherited
inequality implementation exists.  The restoration adds a focused `RegistryId`
test proving equivalent normalized values compare equal and are not unequal,
while distinct canonical values remain unequal.  `ComponentId` has the same
derive boundary; its historical local test module is deliberately disabled with
`#[cfg(any())]`, so it is source-audited rather than claimed as executed test
coverage.

## Restoration scope and evidence

The restoration reinstates the exact upstream `__ne__` algorithm in both
Python bases: invoke `__eq__`, propagate `NotImplemented`, otherwise return
its boolean negation.  It also reinstates the upstream lazy cache access and
the relevant equality/inequality regression matrix, including registered
extensions and direct `NotImplemented` assertions.

The test-first run failed with 32 failures across normalized case and separator
spellings before production code was restored.  The green run is recorded in
the companion plan and closure commit.

## Graham review outcome

The self review found no blocking, important, or style findings. The root
independent Graham review also approved the complete restoration with no
findings. The production change is the
upstream three-branch `__ne__` implementation in each Python base, with no
new synchronization, allocation-bearing hot-path abstraction, logging, or
native interface.  The native test is intentionally one behavior-level
invariant; the upstream Python matrices are retained in full because they
cover two distinct MRO families and runtime extension construction.
