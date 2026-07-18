# AIPerf Flow Language and Module System

## Status

**Proposed**

This design defines Roadmap Plan 3: the complete authoring-language and module
system increment for AIPerf Flow. It covers imports, namespaces, exports,
aliases, integrity-pinned remote imports, cycle handling, bindings and typed
parameters, expressions, formatting and migration, the diagnostic catalog, and
the language-server surface.

It refines:

- [AIPerf Flow Delivery Roadmap, Plan 3](../plans/2026-07-17-aiperf-flow-roadmap.md#plan-3-complete-language-and-module-system);
- [AIPerf Flow Symbol Grammar Implementation Plan](../plans/2026-07-17-aiperf-flow-symbol-grammar.md);
- [AIPerf Flow Design](2026-07-17-aiperf-flow-design.md), especially
  “Document and module system,” “Data and expressions,” and “Compiler
  architecture.”

This is a design specification, not an implementation plan.

The language serves the live-cinematic north star: `.flow` authors semantics,
narration, camera, timeline, interaction, responsive policy, accessibility,
fallback, and quality intent. It never authors Canvas, DOM, SVG, React, or
WebGPU API calls. The compiler lowers one resolved meaning that the Canvas
visual backend, semantic HTML twin, and SVG/HTML fallback consume through shared
evaluated-scene contracts.

## Decision summary

A `.flow` file is one versioned module. A module has a stable canonical URI,
private declarations by default, explicit exports, and imports that bind either
a namespace or selected exported names. Local and package imports resolve under
an explicit build policy. HTTPS imports are allowed only when each import
contains a valid integrity digest and the origin is allowlisted.

The linker builds the complete module graph before type checking or lowering.
Every dependency edge participates in cycle detection. Import cycles are
errors; Flow does not define partially initialized modules.

Values are immutable unless declared as state variables. Expressions are pure,
deterministic, typed, bounded, and incapable of I/O or arbitrary code
execution. State changes occur only through typed registered actions, never as
expression side effects.

The parser produces a lossless syntax tree retaining comments and trivia, while
the typed AST remains the semantic compiler input. The canonical formatter is
syntax-tree based and comment preserving. Migrations are explicit,
version-to-version syntax transformations followed by canonical formatting.

Diagnostics use a stable catalog and one machine-readable schema shared by the
CLI, compiler, migrations, and language server. The language server exposes
diagnostics, completion, hover, definition, references, and rename without
introducing semantics that differ from the command-line compiler.

## Goals

- Make multi-file `.flow` projects deterministic, reviewable, cacheable, and
  safe to build.
- Support reusable Flow-only libraries without generated React, TypeScript,
  JavaScript, or CSS.
- Give names and references one unambiguous meaning across parser, linker,
  compiler, formatter, migration tools, and editor tooling.
- Support typed reusable symbols and declarative definitions without admitting
  arbitrary computation.
- Preserve comments, documentation, and stable source locations through normal
  formatting and migrations.
- Make every failure consumable by humans, CI, AI authoring skills, and LSP
  clients through the same diagnostic contract.
- Lower a fully resolved, type-checked program to Flow IR containing no imports,
  unresolved names, parser trivia, or source-language expressions that are
  eligible for build-time evaluation.
- Preserve stable semantic IDs and timeline anchors across module boundaries so
  pause-to-explore, exact-beat resume, semantic-twin focus, and visual
  continuity cannot drift during linking or symbol expansion.

## Non-goals

- Runtime module loading or runtime network fetching.
- Package installation from `.flow` source.
- JavaScript, TypeScript, CSS, JSX, WebAssembly, shell, or macro execution.
- User-defined imperative functions, recursion, unbounded loops, reflection, or
  dynamic property access.
- A general-purpose programming language or runtime object model.
- A visual editor or a language-server-specific compiler.
- Renderer-specific source constructs or module semantics for Canvas, DOM, SVG,
  React, or WebGPU.
- Backward-compatible interpretation of malformed or ambiguous source.
- Capability package trust and executable extension loading, which remain
  governed by the host capability registry and the later extension-SDK plan.
- Data transforms, event state machines, and bounded runtime streams beyond the
  expression substrate needed to type their future inputs.

## Current executable baseline

The following statements describe the code present when this specification was
written. They are not claims that Plan 3 is complete.

### Present

- `@aiperf/flow-language` uses Chevrotain with recovery enabled.
- The parser emits typed AST nodes with `SourceRange` data.
- Foundation syntax includes a `flow` document, language version, capability
  requirements, token declarations, scenes, a small render vocabulary,
  cameras, timelines, interactions, responsive overrides, narration, reading
  order, and fallbacks.
- Line and block comments are lexed but skipped.
- A canonical formatter exists for the currently parsed AST and is tested for
  semantic idempotence.
- Symbol declarations exist with named typed parameters using simple named type
  references.
- Symbol bodies can contain component invocations.
- Component invocations are accepted in symbol bodies and scene render
  declarations. Their arguments are named and currently accept only string or
  numeric literals and `token(...)` references.
- The compiler has document token tables and per-scene symbol tables, detects
  duplicate local IDs, and validates the foundation reference forms.
- Component invocations lower directly to component IR nodes. Declared symbol
  expansion remains fail-closed for non-empty symbol bodies.
- Diagnostics already carry a code, severity, message, and source range; some
  linker diagnostics also include repair text.

### Proposed by this specification

- Lossless comments and trivia, module imports and exports, namespaces, aliases,
  local/package/remote resolution, integrity policy, dependency manifests, and
  cycle diagnostics.
- A complete lexical-scope and export model.
- Constants, state variables, richer type references, defaulted typed
  parameters, references, objects, lists, maps, unions, and pure expressions.
- Symbol expansion through the same resolved binding model used by ordinary
  declarations and component invocations.
- A stable diagnostic catalog and uniform JSON representation.
- Versioned formatter behavior, comment-preserving migrations, generated
  language references, and an LSP server.

### Relationship to the symbol-grammar plan

The symbol-grammar plan is a deliberately narrow parser/formatter increment. It
defines import stubs, typed symbols, qualified component calls, named slots, and
single-collection `for` loops while explicitly deferring import resolution,
type checking, lowering, and general expressions.

The executable baseline has landed only part of that target: typed symbol
declarations and basic component invocations are present. Import stubs,
qualified calls, slots, collection loops, richer argument values, and their
formatter support are not present. Plan 3 subsumes those deferred semantics.
Where syntax from the earlier plan remains suitable, this design preserves its
intent; this design is authoritative for module identity, binding, type,
expression, diagnostic, and editor semantics.

## Source and module model

### Module boundary

Each `.flow` file defines exactly one module and contains exactly one `flow`
document envelope. The envelope's `language` declaration selects source
language semantics. The document `id` is authored content identity; it is not
the module's resolver identity.

A module is identified internally by a canonical URI:

- local files use normalized absolute `file:` URIs;
- package modules use a canonical package name, resolved package version, and
  package-relative path;
- remote modules use their final normalized HTTPS URL plus verified content
  digest.

Two import spellings that resolve to the same canonical URI refer to one module
instance. Symlink handling is a host policy and must be consistent for an
entire compilation.

### Declaration privacy

Top-level declarations are private by default. Exporting is explicit. Private
names may be used anywhere in the declaring module but cannot be imported or
re-exported.

The exportable declaration categories in Plan 3 are:

- `const`;
- `symbol`;
- typed declarative definitions already supported by the language, such as
  themes, layouts, timelines, interactions, and data transforms as those
  declaration kinds become available.

State `var` declarations are not exportable. Exporting mutable state would make
module evaluation order observable and is rejected.

Scenes are private by default and may be exported only when the host command is
building a library or composing a multi-module document. A normal application
entry module selects its reachable scenes explicitly; importing a module never
adds scenes through side effects.

### Surface syntax

The canonical forms are:

```text
flow "Queue library" as queue-library {
  language 1

  import "./core.flow" as core
  import { SemanticEntity, Queue as QueueView } from "@aiperf/flow-stdlib/viz"
  import "https://cdn.example.com/flow/metrics.flow" as metrics
    integrity "sha256-Px8f4bYQ7L8r8YhY8jXvLxg6D3p1rKfN9wYtQfQmX1I="

  export const defaultCapacity: int = 32

  export symbol RequestQueue(
    id: EntityId,
    capacity: int = defaultCapacity,
    accent: color = core.defaultAccent
  ) {
    QueueView(id = id, capacity = capacity, accent = accent)
  }

  export { RequestQueue as Queue }
}
```

Supported import forms are:

```text
import "./module.flow" as namespace
import { Name, Other as LocalName } from "./module.flow"
```

Supported export forms are:

```text
export const name: Type = expression
export symbol Name(params...) { ... }
export { LocalName, Other as PublicName }
export { Name, Other as PublicName } from "./module.flow"
```

Wildcard imports and wildcard re-exports are not supported. They make public
APIs change when a dependency adds an unrelated export and make rename and
review less reliable.

`namespace` is not a declaration keyword in Plan 3. A namespace is a binding
introduced by `import ... as ...`. This keeps namespaces tied to module
boundaries and avoids merging independently declared namespace blocks.

### Name forms and aliases

A simple name resolves through lexical scopes. A qualified name begins with a
namespace import alias and then names one exported member:

```text
core.SemanticEntity
metrics.defaultScale
```

Plan 3 permits one module qualification boundary. Nested dotted names after the
exported member remain property or domain-specific paths only where the
receiving grammar explicitly permits them; they are not recursively resolved
module namespaces.

Aliases create ordinary local bindings. Imported names never retain a second
hidden unaliased spelling. An alias cannot shadow another binding in the same
scope.

Names are case sensitive. Public symbols and named types use PascalCase.
Bindings, parameters, import aliases, and properties use lower camel case or
the existing accepted kebab-case where a domain grammar requires it. The
formatter preserves spelling but diagnostics enforce the applicable naming
profile.

## Resolution and dependency policy

### Resolver classes

Import specifiers are classified before resolution:

1. `./` and `../` are local file imports relative to the importing module.
2. allowlisted logical roots such as `@aiperf/flow-stdlib/...` are package
   imports resolved by the host's immutable package map.
3. `https://` is a remote import and requires inline integrity.
4. absolute filesystem paths, `file:` source spellings, bare unconfigured
   package names, `http://`, protocol-relative URLs, and all other schemes are
   rejected.

Resolution does not probe alternative extensions or directory indexes.
Specifiers name the exact `.flow` module. Package maps may expose an exact
logical path as an entry point, but that mapping is explicit.

### Root confinement

Local imports must remain within one of the configured source roots. Normalized
paths that escape every root are rejected, including escapes through symlinks
under a policy that resolves real paths. This rule applies before any source is
read into the compilation graph.

Package imports resolve only through the build's package map. A `.flow` file
cannot mutate that map or request package installation.

### Remote imports

Remote imports are build-time source dependencies, not runtime fetches. Every
remote edge must:

- use HTTPS;
- name an allowlisted origin;
- include an `integrity` attribute using an approved Subresource Integrity
  algorithm;
- match the fetched bytes after transfer decoding and before text decoding;
- satisfy configured response-size and redirect limits;
- resolve redirects only to allowlisted HTTPS origins;
- be cached by digest, never trusted merely by URL;
- participate in offline and frozen-build policy.

Plan 3 requires `sha256` support and permits a host to support stronger SRI
algorithms. Unsupported or malformed integrity metadata is an error.

The integrity digest pins the exact module bytes. Imports within a remote module
must independently satisfy the same policy; trust does not flow transitively
from the parent module.

Production and CI builds use frozen resolution: every package and remote module
must already match the dependency manifest. An explicit dependency-refresh
operation may fetch and rewrite the manifest. Ordinary `check` and `build`
commands do not silently update pins.

### Dependency manifest

A successful graph resolution produces a deterministic dependency manifest
containing, for every reachable module:

- canonical URI;
- source kind: entry, local, package, or remote;
- content digest;
- declared language version;
- direct dependency canonical URIs;
- package version or remote integrity metadata when applicable.

Entries and dependency lists are sorted by canonical URI. Absolute workstation
paths are excluded from portable build artifacts; local modules are recorded
relative to the configured source root plus their digest.

The dependency manifest contributes to the build content hash and is retained
in normal and packed build metadata. Flow IR itself contains no unresolved
imports.

### Graph construction

The resolver discovers and parses the complete reachable graph before linking.
The graph is keyed by canonical URI and records source ranges for every import
edge. The same module is parsed once per compilation, even when imported under
multiple aliases.

Export tables are constructed after parsing and before imported-name
resolution. Re-export chains resolve to the originating declaration while
retaining each re-export source location for diagnostics and editor
navigation.

### Cycles

All module dependency cycles are errors in Plan 3. Flow has no type-only import
form and no partially initialized module state, so accepting selected cycles
would add complexity without a valid semantic need.

Cycle diagnostics report:

- one primary diagnostic on the edge that closes the cycle;
- related locations for every edge in order;
- the shortest cycle containing that edge, with ties ordered
  lexicographically by canonical URI;
- a repair suggesting extraction of shared declarations into an acyclic
  module.

The linker may continue analyzing acyclic components to report independent
errors, but no IR is emitted while any reachable cycle exists.

## Scopes and binding resolution

### Scope hierarchy

Scopes are nested in this order:

1. module scope;
2. declarative definition scope;
3. symbol parameter scope;
4. slot or bounded-iteration scope;
5. expression-local scope introduced by a future transform grammar.

Module scope contains local top-level declarations, import namespaces, and
selected imports. Definition scopes contain parameters and local immutable
bindings where the declaration grammar permits them. State variables belong to
the owning scene or interaction state scope, never to module initialization.

### Resolution order

For a simple name, resolution walks lexical scopes from innermost to outermost.
There are no implicit global user declarations. Built-in types and expression
functions occupy separate namespaces and cannot be shadowed by user values.

A qualified name resolves only through its namespace import binding. The linker
does not reinterpret an unresolved dotted name as a component capability ID.
Capability names continue to resolve through capability descriptors in the
grammar positions that explicitly expect capabilities.

Duplicate declarations, conflicting imports, aliases that collide with local
declarations, and duplicate public export names are errors. Shadowing an outer
local value is permitted only inside a bounded iteration or slot parameter and
produces a configurable warning; parameters may not shadow imports or
top-level declarations.

### Stable symbol identity

Each resolved declaration receives a compiler identity derived from:

- canonical module URI;
- declaration kind;
- authored declaration name.

Aliases do not change identity. Re-exports preserve the originating identity.
Source maps retain both the use location and definition location. Renaming a
declaration changes source identity but must not rewrite explicit semantic IDs
authored inside its body.

## Type system

### Built-in value types

Plan 3 defines:

- `string`, `boolean`, `int`, and finite `number`;
- domain scalar types registered by the language or capability descriptors,
  including IDs, colors, durations, units, and references;
- nullable types using `T?`;
- lists using `T[]`;
- objects using named structural fields;
- maps using `map<K, V>`, where `K` is `string`, an enum, or another finite
  scalar key type;
- unions using `A | B`;
- references using `ref<T>`.

`any`, implicit dynamic values, non-finite numbers, and implicit
string-to-number or number-to-string coercions are not supported.

Objects are closed by default: unknown fields are errors. A descriptor may
declare an explicit map-valued extension field rather than opening the complete
object shape.

### Assignability

Assignability is structural for objects and nominal for registered domain
scalars, enums, component contracts, and reference target types. `int` is
assignable to `number`; the reverse requires a registered explicit conversion.
`T` is assignable to `T?`. Union assignment succeeds when the value is
assignable to at least one member.

Union narrowing is explicit through deterministic predicates such as
`value is Type` or descriptor-provided tagged-object discriminants. Truthiness
does not perform type narrowing.

### Constants

Constants use:

```text
const defaultCapacity: int = 32
const queueStyle = { fill: "#1b2430", stroke: token(accent) }
```

`const` is immutable and must have an initializer. Its type may be explicit or
inferred from the initializer. Public constants require an explicit type so
their API is not changed accidentally by editing an initializer.

Constants may refer only to earlier constants in the same module, imported
constants, parameters in an enclosing definition, and pure expression
functions. Module constants cannot depend on scene state.

The compiler builds a constant dependency graph, rejects cycles, evaluates
build-time constants deterministically, and substitutes their typed values
during lowering when source-map fidelity can be retained.

### State variables

Variables use:

```text
var selected: EntityId? = null
var comparisonMode: ComparisonMode = sideBySide
```

`var` declares serializable viewer state owned by a scene or typed interaction
state block. It is not a general mutable local and is forbidden at module
scope, inside symbols, and inside constant initializers.

An initializer is required and must be a build-time expression assignable to
the declared type. A variable can change only through a registered typed action
whose descriptor declares the state write. Expressions can read in-scope state
but cannot assign, mutate objects, or invoke actions.

State variable identity is stable within its owning semantic scope. It is
included in normal IR when runtime evaluation is required and is subject to the
runtime's deterministic serialization and reset policy.

### Typed parameters

Parameters use:

```text
symbol Queue(
  id: EntityId,
  entries: QueueEntry[],
  capacity: int = 32,
  accent: color = token(accent)
) { ... }
```

Required parameters precede defaulted parameters. Names are unique. Defaults
may refer to module constants, imported constants, and earlier parameters, but
not later parameters or runtime state.

Arguments are named. Unknown, duplicate, missing required, or type-incompatible
arguments are errors. The compiler applies defaults before symbol expansion.
Public symbol parameters form an exported API and always require explicit
types.

### References

`ref<T>` is a typed semantic reference, not a string alias. Reference literals
and resolved authored IDs retain target kind information through checking and
lowering. A plain `string` is never implicitly assignable to `ref<T>`.

References may cross modules only to exported declarations. References to
scene-local entities remain within the composed scene scope unless a
declaration contract explicitly exposes a typed reference parameter or result.

## Expression language

### Principles

Expressions are:

- pure and side-effect free;
- deterministic for identical typed inputs;
- bounded in work and allocation;
- serializable when retained for runtime evaluation;
- incapable of filesystem, network, environment, clock, randomness, DOM, or
  capability-registry access except through explicit typed input values;
- evaluated by compiler and runtime implementations with conformance-tested
  semantics.

### Core syntax

Plan 3 supports:

- literals: strings, booleans, finite numbers, `null`, lists, and objects;
- binding and qualified-name references;
- member access on statically known object fields;
- map lookup through a checked `get` function rather than dynamic `value[key]`
  syntax;
- unary `!` and numeric negation;
- arithmetic `+`, `-`, `*`, `/`, and `%`;
- comparisons `==`, `!=`, `<`, `<=`, `>`, and `>=`;
- boolean `&&` and `||`;
- null coalescing `??`;
- conditional `condition ? whenTrue : whenFalse`;
- calls to registered pure expression functions;
- typed reference construction in grammar positions that know the target type.

Operator precedence follows conventional arithmetic, comparison, boolean,
coalescing, and conditional order. Parentheses override precedence. Equality
does not coerce types.

String interpolation, assignment expressions, increment/decrement, arbitrary
method calls, computed member access, lambdas, recursion, and exception
handling are excluded.

### Function registry

Expression functions come from the closed language and capability descriptor
registries. Each descriptor declares:

- qualified function name and version;
- parameter and result types;
- build-time and runtime eligibility;
- determinism and finite-number behavior;
- cost bound as a function of input cardinality;
- evaluator conformance fixtures;
- diagnostic behavior for invalid domains.

There is no reflective function lookup. Unknown functions fail closed.

### Evaluation phases

The type checker classifies each expression as:

- **build-time:** depends only on literals, constants, parameters whose values
  are known during expansion, and build-time functions;
- **runtime:** depends on viewer state, responsive inputs, bounded data, or
  runtime-eligible functions;
- **invalid:** depends on unavailable values or a function not eligible in the
  required phase.

Build-time expressions are evaluated and lowered to values. Runtime expressions
lower to a versioned typed expression IR, not source text. Renderers never parse
or evaluate `.flow` source.

### Bounded declarative iteration

The single-collection `for item in collection` form from the symbol-grammar
plan is retained as declarative expansion, not general control flow. Its
collection must be a statically typed finite list or bounded runtime
collection. Iteration order is the collection's defined order. Map iteration is
forbidden unless keys are explicitly sorted by a registered deterministic
operation.

Nested iteration is permitted only when the compiler can prove a configured
cardinality bound. Otherwise it is a compile error. Iteration cannot contain
state mutation or actions.

## Linking, checking, and lowering

The Plan 3 pipeline is:

```text
source modules
  → lossless lexing and parsing
  → module graph resolution and integrity verification
  → export-table construction and cycle detection
  → lexical name and reference resolution
  → type checking and expression phase classification
  → constant evaluation and declarative symbol expansion
  → semantic validation
  → explicit Flow IR plus dependency manifest and source maps
```

Linking and checking operate on resolved declaration identities, not raw name
strings. Symbol expansion is hygienic: parameters and local iteration bindings
cannot capture names at an invocation site. Generated internal IDs derive from
the invocation identity and declaration-local path, while authored semantic IDs
remain unchanged.

Lowering must remove:

- import and export declarations;
- aliases and namespace bindings;
- private declarations unreachable from entry scenes or exported library
  roots;
- build-time-only constants;
- formatter trivia;
- source-language expression syntax.

Lowering retains:

- explicit runtime expression IR where required;
- dependency and capability requirements;
- declaration/use source-map chains;
- stable semantic IDs and resolved typed references;
- provenance needed by inspection, diagnostics, and packed output.

Normal and packed Flow IR must preserve equivalent module-expanded semantics.

## Lossless syntax and formatter

### Syntax representation

The lexer retains whitespace, line comments, block comments, token text, and
exact ranges. Parsing produces:

- a lossless concrete syntax representation suitable for formatting,
  migrations, and editor edits;
- a typed semantic AST suitable for linking and checking;
- stable syntax-node identities within one parse snapshot;
- mappings between syntax nodes, semantic nodes, and source ranges.

Comments are not semantic compiler inputs except documentation comments when a
declaration explicitly supports generated documentation.

### Comment attachment

The formatter attaches comments by syntax position:

- a same-line comment remains trailing on the preceding construct;
- a comment immediately before a declaration remains leading for that
  declaration;
- comments between list entries remain with the following entry unless they
  trail the previous entry on the same line;
- detached comments separated by a blank line remain detached;
- comments inside empty blocks remain inside those blocks.

Formatting must never silently discard or duplicate comments.

### Canonical formatting

The formatter:

- is deterministic and idempotent;
- emits the selected language version's canonical syntax;
- preserves declaration order because order can aid narrative review, while
  sorting only grammar-defined unordered generated output;
- uses two-space indentation and one final newline;
- uses multiline forms when a declaration or argument list exceeds the
  configured canonical width;
- never resolves imports, changes aliases, rewrites public APIs, evaluates
  expressions, or performs semantic migration;
- refuses to format a source when recovery cannot produce a structurally safe
  syntax tree, returning diagnostics instead of destructive output.

`format --check` compares bytes with canonical output and performs no writes.
Machine-generated edits use minimal text edits where possible, but whole-file
format output remains the canonical oracle.

## Language-version migration

Language migrations are explicit adjacent-version transformations:

```text
aiperf-flow migrate source.flow --to 2
```

Each migration:

- consumes a lossless syntax tree for version `N`;
- applies a registered `N → N+1` transformation;
- preserves comments and source provenance;
- reports every semantic choice it cannot make safely;
- updates the language declaration only after all required transforms succeed;
- formats using the target version's canonical formatter;
- is deterministic and idempotent.

Multi-version migration composes adjacent transformations. Downgrades are not
guaranteed; a supported downgrade must have its own explicit migration.

Migrations do not fetch dependencies, update integrity pins, rename user public
APIs without a language-mandated rewrite, or alter behavior merely to satisfy a
new warning. A migration that cannot preserve semantics fails without writing
the source unless the user explicitly requests a partial patch artifact.

Migration diagnostics and edit sets use the same schema as compiler and LSP
diagnostics. The CLI can emit a machine-readable change report containing old
and new language versions, applied migration IDs, changed ranges, and remaining
diagnostics.

## Diagnostic catalog

### Contract

Every diagnostic contains:

- stable `code`;
- `severity`: `error`, `warning`, `information`, or `hint`;
- concise `message`;
- primary `range`;
- optional actionable `repair`;
- optional `relatedInformation` entries with message and range;
- optional deterministic text edits;
- optional `tags`, including `deprecated` and `unnecessary`;
- source stage and active profile;
- catalog documentation URI derived from the code.

JSON output is versioned independently from prose messages. Consumers key on
the diagnostic code and structured fields, never exact English text.
Diagnostics are sorted by canonical module URI, start offset, severity, then
code.

### Code families

Stable code families are:

- `LEX_*` — invalid characters, unterminated literals, and lexical limits;
- `PARSE_*` — unexpected or missing syntax and recovery limits;
- `MODULE_*` — invalid specifiers, root escapes, missing modules, policy
  violations, integrity failures, manifest mismatch, and cycles;
- `EXPORT_*` — unknown, private, duplicate, or invalid exports and re-exports;
- `LINK_*` — duplicate bindings, unknown names, ambiguity, and invalid
  references;
- `TYPE_*` — unknown types, assignability, parameter, object, map, union, and
  phase errors;
- `EXPR_*` — invalid operators, unknown functions, non-finite results, forbidden
  effects, and exceeded cost bounds;
- `CONST_*` — dependency cycles and non-build-time initializers;
- `STATE_*` — illegal scope, initializer, export, or mutation;
- `SYMBOL_*` — invalid contracts, expansion, hygiene, slots, and cardinality;
- `FORMAT_*` — unsafe recovery and unsupported syntax-version formatting;
- `MIGRATE_*` — unavailable path, unsafe transform, and partial migration;
- `CAPABILITY_*`, `ACCESSIBILITY_*`, `ASSET_*`, and `BUDGET_*` — existing and
  future semantic validation domains.

Codes are never reassigned to a different meaning. A retired code remains
reserved. Message improvements do not require a new code; changed triggering
semantics do.

### Required module diagnostics

Plan 3 defines at least these stable meanings:

- `MODULE_INVALID_SPECIFIER`;
- `MODULE_OUTSIDE_SOURCE_ROOT`;
- `MODULE_NOT_FOUND`;
- `MODULE_REMOTE_ORIGIN_DENIED`;
- `MODULE_INTEGRITY_REQUIRED`;
- `MODULE_INTEGRITY_INVALID`;
- `MODULE_INTEGRITY_MISMATCH`;
- `MODULE_MANIFEST_MISMATCH`;
- `MODULE_IMPORT_CYCLE`;
- `EXPORT_UNKNOWN_NAME`;
- `EXPORT_PRIVATE_NAME`;
- `EXPORT_DUPLICATE_NAME`;
- `LINK_DUPLICATE_BINDING`;
- `LINK_UNKNOWN_NAME`;
- `LINK_UNKNOWN_NAMESPACE_MEMBER`;
- `TYPE_UNKNOWN_TYPE`;
- `TYPE_MISMATCH`;
- `TYPE_MISSING_ARGUMENT`;
- `TYPE_UNKNOWN_ARGUMENT`;
- `TYPE_DUPLICATE_ARGUMENT`;
- `CONST_DEPENDENCY_CYCLE`;
- `STATE_INVALID_SCOPE`;
- `EXPR_UNKNOWN_FUNCTION`;
- `EXPR_PHASE_VIOLATION`;
- `EXPR_COST_LIMIT_EXCEEDED`.

Existing broad parser and linker codes remain valid until a cataloged,
more-specific replacement is introduced through a documented diagnostic-schema
version. Tools may provide both a legacy compatibility code and a specific code
during one transition, but must not emit two user-visible diagnostics for one
failure.

## Language-server surface

### Architecture

The language server is a thin incremental host over the same parser, resolver,
linker, type checker, descriptor registry, diagnostic catalog, and formatter
used by the CLI. It does not contain alternate grammar or type rules.

An editor workspace supplies:

- source roots;
- package map and dependency manifest;
- remote-import allow policy;
- capability descriptor set;
- active validation profile.

The server never performs unprompted network fetches. Remote modules are read
from the verified cache. A user-initiated dependency refresh remains a CLI or
explicit workspace command.

### Diagnostics

Text-document diagnostics include lexical, parse, local binding, type, and
semantic errors available from the current snapshot. Workspace diagnostics add
module resolution, integrity-cache, export, and cycle results.

Diagnostics are published only for the latest document version. Related
locations may point into imported local, package, or cached remote modules.
Remote and package documents are read-only virtual documents.

### Completion

Completion is context and type aware for:

- keywords and declaration forms;
- local bindings and parameters;
- import aliases and exported namespace members;
- named imports and re-exports;
- component and symbol names;
- named arguments not already supplied;
- object fields, enum members, union discriminants, and reference targets;
- registered pure expression functions;
- capability identifiers and descriptor-defined values.

Completion does not suggest private remote names, unavailable capabilities,
out-of-scope state, or incompatible values.

### Hover

Hover displays:

- declaration kind and canonical typed signature;
- inferred or declared type;
- constant value when safely build-time evaluated;
- documentation comments and descriptor documentation;
- defining module and public export path;
- capability version and deprecation information;
- remote digest for imported remote declarations.

Hover does not execute runtime expressions.

### Definition and references

Go-to-definition follows local bindings, imports, aliases, namespace members,
re-exports, parameters, references, and descriptor-backed capability names.
For re-exports, clients may navigate first to the public re-export or directly
to the origin; the server exposes both links.

Find-references is identity based, not text based. It distinguishes shadowed
bindings and aliases and can operate across the configured workspace and
verified dependency graph.

### Rename

Rename is supported for user declarations, parameters, local aliases, and
semantic IDs whose reference domain has a complete typed index.

Rename:

- updates definitions, imports, re-exports, qualified uses, and references
  identified by binding identity;
- checks collisions and naming rules before returning edits;
- formats affected syntax ranges after edits;
- never modifies package or remote cached modules;
- refuses a rename that would require changing a read-only dependency;
- treats an exported-name rename as a public API change and requires the client
  to confirm that scope through a change annotation.

String contents and unrelated textual matches are never renamed.

### Formatting and code actions

The server provides document and range formatting through the canonical
formatter. Range formatting expands to the smallest safe syntax construct.

Initial code actions include:

- add a missing named import when exactly one reachable export matches;
- qualify an ambiguous name with an existing namespace alias;
- remove an unused private import or alias;
- add a missing integrity attribute only from a verified dependency-manifest
  entry;
- apply cataloged deterministic diagnostic fixes;
- invoke an explicit language migration preview.

Code actions that change dependency policy, fetch remote content, or choose
among multiple semantic targets are not automatic quick fixes.

## Generated references

Plan 3 generates, from grammar metadata, AST/type descriptors, the diagnostic
catalog, and capability descriptors:

- grammar and syntax reference;
- built-in type and expression-function reference;
- diagnostic-code reference;
- module-resolution and security-policy reference;
- machine-readable schema for editor and authoring-skill consumption;
- capability-aware completion and hover metadata.

Generated references describe the selected language version. They do not infer
features from implementation-only TypeScript types, and generation fails when
the grammar, catalog, and exported descriptor metadata disagree.

## Security and reproducibility

- Source cannot execute code during parsing, linking, formatting, migration, or
  constant evaluation.
- Remote source is pinned by content and constrained by explicit origin,
  redirect, size, and offline policies.
- Import paths cannot escape configured roots.
- Expression and declarative iteration costs are bounded before runtime.
- Module, constant, and symbol expansion cycles fail closed.
- Build hashes include source bytes, dependency manifest, language version,
  capability manifest, and migration-relevant schema versions.
- Diagnostics and generated manifests avoid leaking absolute local paths in
  portable artifacts.
- Normal and packed IR contain no source import instruction that could trigger
  browser network access.

## Verification requirements

The design is complete when executable conformance covers:

- local, package, named, namespace, aliased, and re-export resolution;
- canonical-URI deduplication, root confinement, exact-path resolution, and
  deterministic dependency manifests;
- allowed and denied remote origins, redirects, malformed integrity, digest
  mismatch, frozen and offline operation, and transitive remote imports;
- deterministic shortest-cycle diagnostics with related import locations;
- privacy, duplicate exports, alias collisions, shadowing, and identity-based
  reference resolution;
- constants, constant cycles, state scope, typed defaults, objects, maps,
  unions, references, and argument checking;
- operator precedence, phase classification, finite-number policy, function
  registry closure, and expression cost limits;
- hygienic symbol expansion and bounded declarative iteration;
- comment preservation, formatter byte idempotence, parser/formatter semantic
  round trips, and unsafe-recovery refusal;
- every migration path, migration idempotence, comment retention, and
  no-write-on-failure behavior;
- one malformed fixture per stable diagnostic code and stable JSON snapshots;
- LSP completion, hover, definition, references, rename, diagnostics,
  formatting, stale-version suppression, and read-only dependency behavior;
- equivalent semantics and source provenance in normal and packed Flow IR;
- deterministic output and content hashes across repeated clean builds.

Property-based tests cover formatter idempotence, expression parsing and
precedence, import-graph traversal order, alias-independent declaration
identity, and comment attachment. Fuzzing covers lexer/parser recovery,
migration input, integrity metadata, and module graph shapes within configured
resource limits.

## Compatibility and rollout boundary

The source language version gates all new syntax. Existing valid foundation
documents remain valid under their declared version and retain their current
meaning. Plan 3 may make previously ignored or unsupported constructs valid
only through an explicit language-version rule or migration.

The current `token` declaration remains accepted as a domain declaration. It
may be represented internally through the constant/type substrate, but its
surface spelling and existing semantics do not change in this design.

The current formatter's output for unchanged foundation constructs remains
canonical unless a versioned formatter rule explicitly changes it. Lossless
trivia support adds comment preservation; it does not make whitespace
semantically significant.

Plan 3 ends at a fully linked, type-checked, module-expanded language and its
tooling surface. Later roadmap plans add the broad layout, visual, narrative,
data, interaction, and extension vocabularies on top of these contracts rather
than defining competing module, expression, diagnostic, or LSP semantics.
