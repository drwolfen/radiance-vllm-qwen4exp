# R9V vLLM fork provenance

This fork retains vLLM's Apache License 2.0. The Qwen3.8 Flash Next model and
PLE foundation descend from commits submitted through vLLM pull request
`vllm-project/vllm#53896`, including `d4d0f73ef171154eac6f1914dca47001d662cfbb`
and `d8d2b86cb88c91bbfad7fde09271d20147b8d50c`. R9V's ROCm, GGUF, offload,
profiling, and kernel integration is carried in R9V-authored commits and files
marked as modifications.

The dual-card deployment was developed alongside the Radiance workflow.
Radiance's unlicensed launcher and R4D source are not included in this fork.
The multimodal draft-mask alignment behavior is implemented here from its
input/output invariant and covered by R9V tests; it is not a copy of the
historical Radiance patch text.
