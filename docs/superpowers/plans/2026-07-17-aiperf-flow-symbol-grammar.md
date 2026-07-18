# AIPerf Flow Symbol Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve and extend the landed symbol-expansion pipeline that
validates symbol-call props, substitutes parameter references, recursively
expands local symbol invocations, and rejects recursion or unsupported forms
with source-mapped diagnostics.

**Architecture:** `compileSource` parses, collects symbols, expands the parsed
document, links the expanded document, validates, lowers, and schema-validates.
`expandSymbolInvocations` is an immutable recursive parsed-document
transformation with parameter substitution, strict bindings, leaf preservation,
and direct/indirect recursion diagnostics. Linking remains downstream so it
sees only the expanded document.

**Tech Stack:** TypeScript strict mode, Chevrotain 12, Vitest 4, `@aiperf/flow-language`, `@aiperf/flow-schema`.

## Current state and execution status

- `@aiperf/flow-language` already exports `SymbolDefinitionAst`,
  `ComponentInvocationAst`, `ParamDeclarationAst`, and
  `PropAssignmentAst`.
- The parser already emits document-level symbol definitions, component
  invocations in symbol bodies, and component invocations in scene render
  declarations. The formatter already round-trips both forms.
- `collectSymbols(document)` already builds a `SymbolTable` and reports
  `SYMBOL_DUPLICATE_EXPORT`.
- `validateProps(props, schema, range)` already reports unknown, missing, and
  type-mismatched props against a strict component-shaped schema.
- `expandSymbolInvocations(document, symbols)` implements recursive expansion,
  parameter substitution, immutable cloning, strict binding checks, leaf
  preservation, and recursion detection.
- `compileSource` runs parse → collect symbols → expand → link → validate →
  lower → schema-validate.
- `SYMBOL_EXPANSION_UNSUPPORTED` remains only for genuinely unsupported symbol
  forms such as named slots; it is not the default for non-empty bodies.
- Tasks 1–4 are landed and tested. Their original failing-test expectations
  document the TDD sequence and must not be recreated by reverting the
  implementation.

## Global Constraints

- Modify only `apps/aiperf-flow/packages/language/`,
  `apps/aiperf-flow/packages/compiler/`, and this plan when executing this
  increment.
- `apps/aiperf-flow/preview/**` is forbidden.
- Preserve the existing `.flow` syntax for symbol definitions and component
  invocations.
- Preserve existing foundation parsing, formatting, compilation, diagnostics,
  and deterministic IR.
- A component invocation whose name matches a local symbol is a symbol call. An
  invocation whose name is absent from the local `SymbolTable` remains a leaf
  capability invocation.
- Expand symbol calls before linking, semantic validation, and lowering.
  Neither linker, `validate`, nor `lower` should learn symbol semantics.
- Expansion is immutable and deterministic. Do not mutate parsed AST nodes,
  symbol definitions, linked tables, or input prop arrays.
- Preserve declaration source maps on cloned body invocations and call-site
  source maps on substituted argument values.
- Reuse `collectSymbols` and `validateProps`; do not create parallel symbol
  tables or duplicate strict-prop validation.
- Detect direct and indirect recursive symbol calls and fail closed with a
  source-mapped diagnostic.
- Named slots and `for` loops are deferred tasks. Do not add slot AST, loop AST,
  parser rules, formatter branches, expansion behavior, or lowering behavior
  in this increment.
- Import resolution, namespaces, defaults, optional params, arrays, object
  expressions, and general expression evaluation remain out of scope.
- Activate `.venv` before repository commands:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Do not create git commits unless the user explicitly requests them.

---

## File map

| File | Responsibility |
|---|---|
| `apps/aiperf-flow/packages/language/src/ast.ts` | Add a parameter-reference value AST and include it in authored prop values |
| `apps/aiperf-flow/packages/language/src/parser.ts` | Parse bare lower-case identifiers as parameter references in prop values |
| `apps/aiperf-flow/packages/language/src/formatter.ts` | Format parameter references without quotes |
| `apps/aiperf-flow/packages/language/test/symbol.test.ts` | Prove symbol-body parameter-reference parse and format behavior |
| `apps/aiperf-flow/packages/compiler/src/expand-symbols.ts` | Validate bindings, substitute parameter references, recursively expand local symbols, and diagnose cycles |
| `apps/aiperf-flow/packages/compiler/src/lower.ts` | Enforce the invariant that expansion resolved every parameter reference before lowering |
| `apps/aiperf-flow/packages/compiler/src/index.ts` | Preserve symbol collection and expansion before linking and validation |
| `apps/aiperf-flow/packages/compiler/test/expand-symbols.test.ts` | Unit-test expansion, substitution, leaf preservation, diagnostics, and immutability |
| `apps/aiperf-flow/packages/compiler/test/compile.test.ts` | Prove the complete compiler pipeline expands symbols before lowering |

---

### Task 1: Parameter references in existing symbol bodies

**Files:**
- Modify: `apps/aiperf-flow/packages/language/src/ast.ts`
- Modify: `apps/aiperf-flow/packages/language/src/parser.ts`
- Modify: `apps/aiperf-flow/packages/language/src/formatter.ts`
- Test: `apps/aiperf-flow/packages/language/test/symbol.test.ts`

**Interfaces:**
- Produces: `ParameterReferenceAst`.
- Changes: `PropAssignmentAst.value` from `ValueAst` to
  `ValueAst | ParameterReferenceAst`.
- Preserves: existing literal and `token(...)` prop values.

- [ ] **Step 1: Add a failing parse-and-format test**

Add a test whose source contains:

```text
symbol Wrapper(id: string, label: string) {
  SemanticEntity(id = id, label = label)
}
```

Assert that both prop values have:

```typescript
{ kind: "parameter-reference", name: "id" }
{ kind: "parameter-reference", name: "label" }
```

Also assert that `formatDocument` emits
`SemanticEntity(id = id, label = label)` and reparses to the same AST shape
after source maps are removed.

- [ ] **Step 2: Run the focused test and confirm the missing grammar**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-language -- test/symbol.test.ts
```

Expected: FAIL because the current `value` rule accepts only literals and token
references.

- [ ] **Step 3: Add the minimal AST, parser branch, and formatter branch**

Add:

```typescript
export type ParameterReferenceAst = AstNode<"parameter-reference"> &
  Readonly<{ name: string }>;

export type PropValueAst = ValueAst | ParameterReferenceAst;
```

Change `PropAssignmentAst.value` to `PropValueAst`. Parse a bare `Identifier`
as `ParameterReferenceAst` after the existing quoted-string, number, and
`token(...)` alternatives. Format it as `value.name`.

Do not accept qualified names, property access, calls, arrays, objects, slots,
or loops.

- [ ] **Step 4: Run language regression tests and build**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-language
npm run build -w @aiperf/flow-language
```

Expected: PASS, including existing component-invocation and foundation
formatter tests.

---

### Task 2: Recursive symbol invocation expansion

**Files:**
- Modify: `apps/aiperf-flow/packages/compiler/src/expand-symbols.ts`
- Modify: `apps/aiperf-flow/packages/compiler/src/lower.ts`
- Modify: `apps/aiperf-flow/packages/compiler/test/expand-symbols.test.ts`

**Interfaces:**
- Consumes: `LinkedDocument`, `SymbolTable`.
- Produces:
  `expandSymbolInvocations(linked: LinkedDocument, symbols: SymbolTable):
  Result<LinkedDocument>`.
- Reuses: `validateProps`.

- [ ] **Step 1: Replace stub-oriented tests with failing expansion tests**

Cover these cases with small immutable AST fixtures:

1. An empty symbol table returns the same `LinkedDocument` object.
2. A scene invocation absent from the symbol table remains unchanged as a leaf
   capability invocation.
3. A scene invocation matching a symbol is replaced in place by the symbol
   body's invocations.
4. A symbol body parameter reference is replaced by the corresponding call
   prop value.
5. A symbol body may invoke another local symbol; expansion is recursive and
   preserves authored order.
6. Reusing one symbol twice produces independent cloned invocation and prop
   arrays.
7. Direct and indirect recursion report `SYMBOL_EXPANSION_CYCLE`.
8. An undeclared parameter reference reports `SYMBOL_UNKNOWN_PARAMETER`.
9. Unknown, missing, and type-mismatched call props are rejected through
   `validateProps`.
10. The input document, definitions, and props are unchanged after success and
    failure.

- [ ] **Step 2: Run focused tests and confirm the stub fails**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-compiler -- test/expand-symbols.test.ts
```

Expected: FAIL because the stub rejects every non-empty body and accepts an
arbitrary generic document rather than a `LinkedDocument`.

- [ ] **Step 3: Derive strict symbol-call schemas**

For each `SymbolDefinitionAst`, derive a `ComponentPropsSchema` keyed by its
params. All params are required in this increment.

Map built-in type names as follows:

```typescript
const BUILTIN_PARAM_KINDS = {
  string: "string",
  number: "number",
  boolean: "boolean",
  EntityId: "string",
} as const;
```

Treat other nominal type names as `"json"` until the language gains a complete
type system. Resolve literal call props directly and resolve `token(...)`
through `linked.tokens` before calling `validateProps`. Keep a separate map of
the authored `ValueAst` bindings for substitution so token references retain
their source form instead of being replaced by resolved literals.

A parameter reference with no active symbol binding—including one authored
directly on a scene-level leaf capability—is invalid. Report
`SYMBOL_UNKNOWN_PARAMETER` at that value's source map.

- [ ] **Step 4: Implement immutable substitution and recursive expansion**

Use a recursion stack of symbol names, not a global visited set. This allows the
same acyclic symbol to be invoked multiple times while detecting:

```text
A → A
A → B → A
```

For a matching local symbol call:

1. Validate and resolve its call props.
2. Build a binding map by declared parameter name.
3. Clone each body invocation.
4. Replace every `parameter-reference` prop with its bound authored value.
5. Recursively expand each cloned body invocation.
6. Splice the resulting invocations into the enclosing scene or symbol body in
   declaration order.

For a non-matching invocation, clone only when substitution changed one of its
props; otherwise preserve the original object.

Return a new `LinkedDocument` with:

```typescript
{
  ...linked,
  document: {
    ...linked.document,
    scenes: expandedScenes,
  },
}
```

The existing token and scene link tables remain valid because this increment
expands only `ComponentInvocationAst` nodes; component invocations do not
participate in the linker's rect/connector ID tables.

Update `lower.ts`'s exhaustive prop-value handling for the widened language
union. A surviving `parameter-reference` is an internal pipeline invariant
violation and must throw an explicit internal error; lowering must not attempt
to bind or expand it. `compileSource` tests must prove authored failures are
returned as expansion diagnostics before this guard is reachable.

- [ ] **Step 5: Run focused compiler tests**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-compiler -- test/expand-symbols.test.ts test/components.test.ts test/symbols.test.ts
```

Expected: PASS. The obsolete assertion that every non-empty body produces
`SYMBOL_EXPANSION_UNSUPPORTED` is removed.

---

### Task 3: Wire expansion into `compileSource`

**Files:**
- Modify: `apps/aiperf-flow/packages/compiler/src/index.ts`
- Modify: `apps/aiperf-flow/packages/compiler/test/compile.test.ts`

**Interfaces:**
- Consumes: `collectSymbols(parsed.value)`, `link(parsed.value)`,
  `expandSymbolInvocations(linked.value, symbols.value)`.
- Produces pipeline:
  parse → collect symbols → link → expand → validate → lower →
  schema-validate.

- [ ] **Step 1: Add a failing end-to-end compiler test**

Compile source with:

```text
symbol LabeledEntity(id: string, label: string) {
  SemanticEntity(id = id, label = label)
}
```

Invoke `LabeledEntity(id = "cli", label = "CLI")` in a scene. Assert that the
resulting scene root is a component with:

```typescript
{
  capabilityId: "SemanticEntity",
  id: "cli",
  props: { id: "cli", label: "CLI" },
}
```

Also assert that no `LabeledEntity` component reaches IR.

- [ ] **Step 2: Add failing pipeline diagnostic tests**

Through `compileSource`, assert:

- duplicate symbol exports report `SYMBOL_DUPLICATE_EXPORT`;
- recursive symbol calls report `SYMBOL_EXPANSION_CYCLE`;
- bad symbol-call props preserve the diagnostics produced by `validateProps`;
- a non-symbol component invocation still lowers as a capability component.

- [ ] **Step 3: Run the focused compile tests**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-compiler -- test/compile.test.ts
```

Expected: FAIL because `compileSource` does not call `collectSymbols` or
`expandSymbolInvocations`.

- [ ] **Step 4: Update the compiler pipeline**

After parsing, collect symbols and short-circuit on duplicate exports. Link the
parsed document, then call expansion with the linked value and symbol table.
Pass the expanded linked document to `validate`, and pass the validated result
to `lower`.

On success, accumulate diagnostics in stage order:

```text
parse → collect symbols → link → expand → validate → schema validation
```

Do not lower or validate the pre-expansion document.

- [ ] **Step 5: Run compiler tests and build**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-compiler
npm run build -w @aiperf/flow-compiler
```

Expected: PASS.

---

### Task 4: Regression and scope gate

**Files:**
- Verify only: `apps/aiperf-flow/packages/language/`
- Verify only: `apps/aiperf-flow/packages/compiler/`

- [ ] **Step 1: Run language and compiler package gates**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-language
npm test -w @aiperf/flow-compiler
npm run build -w @aiperf/flow-language
npm run build -w @aiperf/flow-compiler
```

Expected: all tests pass and both packages build.

- [ ] **Step 2: Run the workspace gate**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm run flow:check
```

Expected: PASS.

- [ ] **Step 3: Confirm forbidden scope is untouched**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust
git diff --name-only -- apps/aiperf-flow/preview/
```

Expected: no output from this implementation.

---

## Deferred follow-up tasks

- [ ] **Named slots:** define slot declaration/invocation AST, parser and
  formatter syntax, slot binding diagnostics, expansion semantics, and lowering
  contracts in a separate plan.
- [ ] **`for` loops:** define collection/reference types, lexical scope,
  deterministic iteration, expansion limits, diagnostics, parser/formatter
  support, and lowering contracts in a separate plan.

Neither deferred task is a prerequisite for literal/token/parameter-based
symbol expansion in this plan.

## Dependency order

```text
Task 1 → Task 2 → Task 3 → Task 4
```

## Spec coverage self-review

| Requirement | Task |
|---|---|
| Start from existing symbol and invocation AST | Current state, Task 1 |
| Reuse `collectSymbols` | Task 3 |
| Reuse `validateProps` | Task 2 |
| Replace fail-closed expansion stub | Task 2 |
| Expand local symbol invocations | Task 2 |
| Substitute symbol parameters | Tasks 1–2 |
| Wire expansion between link and validate | Task 3 |
| Preserve non-symbol capability invocations | Tasks 2–3 |
| Detect recursive expansion | Tasks 2–3 |
| Defer slots and `for` loops | Global Constraints, Deferred follow-up tasks |
| Keep forbidden preview scope untouched | Global Constraints, Task 4 |

## Execution options

Plan complete and saved to
`docs/superpowers/plans/2026-07-17-aiperf-flow-symbol-grammar.md`.

1. **Subagent-driven (recommended)** — execute one task at a time with review
   between tasks.
2. **Inline execution** — execute Tasks 1–3 in one session, then run Task 4 as
   the final gate.
