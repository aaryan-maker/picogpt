# Notes

Machine for all numbers: Intel i7-10510U (4c/8t), 8 GB RAM, no GPU. NumPy on
OpenBLAS, `OPENBLAS_NUM_THREADS=4`.

## Correctness (vs Hugging Face `transformers`, 22-token prompt)

| path | max \|Δlogit\| | mean \|Δlogit\| | argmax agreement |
|---|---|---|---|
| stateless        | 1.2e-4 | 2.2e-5 | 100% |
| KV-cache         | 1.2e-4 | 2.0e-5 | 100% |
| int8 + KV-cache  | 2.0    | 0.45   | 95.5% (same next token) |

## Perf log (decode throughput, GPT-2 124M)

| # | change | forward T=5 | decode tok/s |
|---|--------|-------------|--------------|
| 0 | naive baseline | ~1500 ms | 0.2 |
| 1 | `OPENBLAS_NUM_THREADS=4` (was defaulting to 24 on a 4-core chip) | ~1400 ms | ~0.9 |
| 2 | pin float32 constants in gelu/layernorm/attention (a Python-float `eps` was promoting every activation to float64) | 133 ms | 5.2 |
| 3 | KV cache + mask hoist + skip mask for single-token decode + `last_only` prefill logits | n/a | ~24 |

## KV cache (`bench.py --new-tokens 64 --reps 4`)

| config | prefill (median) | decode tok/s | first→last token latency |
|---|---|---|---|
| no-cache | 204 ms | 3.0 | 209 ms → 472 ms |
| kv-cache | 197 ms | 24.2 | 41 ms → 41 ms |

Speedup ~8x at 64 tokens (7.7x at 48), widening with length.

## int8 quantization (`--quantize`)

| | fp32 | int8 |
|---|---|---|
| weight bytes | 498 MB | 243 MB |
| decode tok/s | 24.2 | 3.9 |

NumPy has no int8 GEMM, so each matmul still pays a full-matrix `astype(float32)`
before the BLAS call. int8 is a memory lever (avoid paging / fit the reference
model alongside), not a speed one. Opt-in.

## CPU GEMM roofline (measured)

- big/fat fp32 GEMM: ~82 GFLOPS
- skinny GEMM (M=1..5): ~5-12 GFLOPS, memory-bandwidth bound

## Next

1. Fused int8/fp16 matmul kernel (Numba) to make quantization a speed win.
2. Port to Qwen2.5-0.5B (RoPE, RMSNorm, GQA, SwiGLU).
3. Batched prefill / speculative decoding.
4. Quantize `wte` (154 MB, dominates the footprint; it's also the tied output projection).
