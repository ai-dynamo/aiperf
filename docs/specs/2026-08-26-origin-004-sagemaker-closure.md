# Origin 004 SageMaker timestamp closure

Malformed `eventMetadata.inferenceTime` is already handled by the native
loader: `parse_capture` requires a string and maps every ISO-8601 parse failure
to `DatasetError::Validation` with the row origin. The exact upstream fix is in
ancestry; no functional Rust change is needed. Disposition: **already-covered**.
