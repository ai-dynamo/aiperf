# Origin commit 093038afab: top-level JSON array responses

## Finding

The upstream change widens endpoint response auto-detection from JSON objects
to JSON bodies of any shape. Its only coverage is one unit-test file with
three cases: a top-level array of ranking objects, one numeric embedding
vector, and a batch of numeric embedding vectors. There are no upstream
integration or E2E tests.

The native KServe V1 auto-detection path previously rejected non-object JSON
before its existing ranking/embedding probes. This was applicable: native
KServe response data already has `Rankings` and `Embeddings` variants, so the
array behavior belongs in that shared auto-detection seam.

## Specification and implementation

For a top-level JSON array, classify a non-empty array of objects as
`ResponseData::Rankings`; classify a non-empty numeric array as one embedding
vector; classify a non-empty array of numeric arrays as a batch of embeddings.
Empty or mixed-shape arrays remain unrecognized. Object response behavior is
unchanged. The V1 parser now routes non-object JSON through this detector.

## Test audit and closure

Native unit coverage mirrors all three upstream cases. Focused KServe runtime
tests and formatting pass. No separate Rust E2E test is applicable because the
upstream change is an endpoint parser utility with no orchestration or server
fixture change. Graham review found no issues: the diff is localized to the
response parser, avoids synchronization and async changes, and performs only
the bounded response-array traversal already required for classification.
