# Sol Plan: Origin/Main 058 FakeTokenizer closure

## Scope

Close upstream `9b60a3d479fd808373c077686c0bbe367d431bec` by restoring its
lost Python fake-tokenizer compatibility method and proving it has no native
Rust product implementation surface.

## Steps

1. Inspect the exact upstream diff and establish that it changes only
   `tests/harness/fake_tokenizer.py` by returning zero prompt special tokens.
2. Prove exact ancestry through `e23f5f835e05d1f0e1cc8a403ebfa50096feca5c`
   and compare the current Python fake with the upstream parent tree.
3. Write and run a focused Python RED regression for the missing method, then
   restore the exact upstream `return 0` implementation and verify GREEN.
4. Inspect `TextTokenizer` and its native test fixtures: the existing
   `num_special_tokens_to_add() -> 0` default is the native equivalent, while
   real Hugging Face tokenizers derive a nonzero value from their
   post-processor.
5. Record finding/spec/tracker evidence and Graham-review the focused Python
   restoration. Do not add a synthetic Rust fake or redundant zero override.

## Future shared-head integration

This closure branch is based on shared head
`e574adc1b9b087c60e50e5fe346fb71264910111`.  Before integration, create a
fresh standalone worktree from the then-current shared head, merge this
closure tip there, resolve only documentation churn, and inspect the resulting
diff.  The shared branch must receive an actual two-parent `--no-ff` merge of
that reviewed closure tip; it must not attempt to merge `9b60a3d479` again
because the upstream object is already an ancestor.
