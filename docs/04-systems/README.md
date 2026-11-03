# 04 · Systems

Intelligence at scale is an engineering achievement. Frontier models are trained on thousands of accelerators for weeks and served to millions of users. This section covers the hardware and software that make that possible, and the physical limits that motivate the [hardware frontiers](../06-frontiers/README.md).

| Page | What you will learn |
|---|---|
| [Hardware and training at scale](hardware-and-training.md) | GPUs and TPUs, the roofline model, mixed precision, data/tensor/pipeline parallelism, ZeRO |
| [Inference and efficiency](inference-and-efficiency.md) | KV caching, batching, quantization, distillation, mixture-of-experts, speculative decoding |
| [The KV cache: paging, sharing and compression](kv-cache.md) | How big the cache gets and why decoding is memory-bound, paging and prefix sharing (tested code), prefill/decode disaggregation, MQA, GQA and latent caches, token eviction versus query-aware selection, 2- and 3-bit quantization, and what compression costs in multi-turn use |
| [Speculative decoding](speculative-decoding.md) | The draft-and-verify acceptance rule and its exactness proof, the speed-up formula and best draft length (tested code), drafters from small models to Medusa and EAGLE-3, token trees and optimal transport, when batching, long contexts and mixture-of-experts change the answer |
| [Compilers for machine learning](ml-compilers.md) | Fusion and tiling, IRs and lowering, Halide, TVM, XLA, Triton, MLIR, PyTorch 2, learned schedule search, LLM-written kernels |

```mermaid
flowchart LR
    A[Algorithms] <--> H[Hardware]
    H <--> S[Software stack<br/>compilers, kernels, runtimes]
    S <--> A
```

Hooker (2020) calls this co-dependence **the hardware lottery**: research ideas succeed partly because they suit the hardware available, not only because they are better.
