# Origin #15 closure: MMVU documentation memory bound

Upstream commit `2eb04aa2f8` changes only `docs/tutorials/mmvu.md`. It bounds
the example vLLM video endpoint with 16 sampled frames and one active sequence,
then explains the smaller-GPU OOM mechanism and how to raise the limits on
larger GPUs.

This is documentation guidance for the external vLLM launch command. It does
not change AIPerf runtime behavior, configuration, protocol, or artifacts, so
there is no Rust implementation to port. The repository contains no upstream
integration or end-to-end test associated with this documentation-only commit;
the MMVU tutorial remains an operator-run GPU workflow and cannot be validated
in the native test harness without provisioning the external model and GPU.

Disposition: not-applicable; exact merge performed for campaign ancestry.
