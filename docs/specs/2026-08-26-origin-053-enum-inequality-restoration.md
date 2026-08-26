# Origin #53 enum inequality restoration

## Goal

Restore origin/main `e5ebe915df`'s case-insensitive enum equality contract in
the shipped Python product, and prove that Rust's canonical identifier types
already satisfy the corresponding inequality invariant.

## Requirements

1. `CaseInsensitiveStrEnum.__ne__` and `ExtensibleStrEnum.__ne__` must call
   their own `__eq__`, return `NotImplemented` unchanged, and otherwise return
   the negated equality result.
2. Equality and inequality must be complementary for normalized case,
   hyphen/underscore, enum-member, unrelated-string, and unsupported operands.
3. Runtime-registered `ExtensibleStrEnum` members must retain the same
   invariant and their lazy normalization cache behavior.
4. The restoration must preserve the upstream exception-based lazy cache access
   in both enum bases.
5. Native identifiers must remain construction-time canonical strings with
   derived `Eq`/`PartialEq`; no Rust emulation of Python MRO is permitted.

## Non-goals

No public Rust API, wire contract, registry behavior, or dependency changes are
needed.  This is a product restoration for a source delta discarded by an
earlier semantic merge despite the upstream object being an ancestor.

## Verification

- Run the two Python enum unit modules together; they cover both enum bases,
  cross-enum values, registered extensions, string-left comparison, and direct
  `NotImplemented` behavior.
- Run the focused `RegistryId` Rust test with `sccache` and a `/mnt/4tb` target
  directory to establish the native invariant; source-audit `ComponentId`,
  whose historical local test module is intentionally `#[cfg(any())]`.
- Confirm the restored production files are tree-equivalent to upstream
  `e5ebe915df` for the changed enum sections.
