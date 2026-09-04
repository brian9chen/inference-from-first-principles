# Benchmark workloads

Small, reproducible workload definitions will be added as benchmark stages are
implemented. The same workloads will be reused when comparing the educational
runtime with vLLM.

Each workload will record prompt data or its generation seed, model and tokenizer
revision, input and output token counts, sampling settings, batch size,
concurrency, and arrival pattern where applicable. Prompts will be synthetic or
public and will not contain credentials or private data.
