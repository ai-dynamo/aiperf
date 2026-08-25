# Origin #20 finding: empty `raw_messages` delta

Upstream `446d2cd4b3` changes the Python endpoint merge contract so
`raw_messages=[]` is an explicit zero-message delta. It must be skipped rather
than synthesized into a `content: []` message, while `raw_messages=None` keeps
the structured-turn synthesis path. Reset-context on an empty delta still
clears accumulated messages. The upstream commit has focused unit tests across
chat, Responses, dataset validation, and worker cache-bust injection, but no
integration or end-to-end tests.

The native Rust endpoint model already represents this distinction as
`Option<Vec<Value>>`, but dataset resolution currently converts an explicit
empty raw-message segment into `None`, and the endpoint lowerers use
`!raw_messages.is_empty()` before deciding whether to synthesize. This is an
applicable native port. The implementation will preserve explicit presence at
resolution and make all message-array lowerers treat `Some(empty)` as a
no-op, with focused unit coverage for chat/Responses and reset behavior.

Disposition: `applicable`.
