# wilupgu

A small GPU compute-graph and tensor library for Rust, built to back a
from-scratch transformer training pipeline ([akasha-core](https://github.com/2j87/akasha-core)).
It exposes a single tensor/shader/graph abstraction over **two interchangeable
GPU backends**:

- **Vulkan** (via [`wgpu`](https://github.com/gfx-rs/wgpu)) — the default, cross-platform backend. Shaders are written in WGSL.
- **CUDA** (via [`cudarc`](https://github.com/coreylowman/cudarc), optional, behind the `cuda` feature) — NVIDIA-only, using cuBLAS for matrix multiplies and NVRTC-compiled CUDA C for everything else.

Both backends implement the *same* set of operations (forward and backward
passes for a transformer block, plus AdamW), so model code written against
`wilupgu`'s `Tensor`/`ComputeGraph` API runs unmodified on either backend —
switching is a Cargo feature flag, not a code change.

## Why two backends

Vulkan/wgpu is the safe default: it works on any GPU (NVIDIA, AMD, integrated)
without extra system dependencies. CUDA is opt-in for NVIDIA hardware where
cuBLAS's hand-tuned GEMM kernels give a meaningful speedup over generic
compute-shader matrix multiplication — in practice, on an RTX 4050 training a
162M-parameter transformer, CUDA ran at **~80 steps/min vs Vulkan's ~31
steps/min** for the same model and workload (roughly 2.5x faster).

## Building

```bash
# Vulkan only (default, no extra system deps beyond a Vulkan driver)
cargo build --release

# With CUDA support (requires the NVIDIA CUDA Toolkit installed; this crate
# was developed/tested against CUDA 13.3)
cargo build --release --features cuda
```

The CUDA feature pulls in `cudarc` with `cublas` + `nvrtc` +
`cuda-version-from-build-system`. At runtime, code that asks for a CUDA
context (`WgpuContext::new_cuda()`) will fail gracefully (return `None`) if no
compatible NVIDIA GPU/driver is found, so a `cuda`-featured binary can still
fall back to Vulkan at runtime if desired.

## Core abstractions

- **`Tensor`** — a backend-tagged GPU buffer (`TensorStorage::Wgpu` or
  `TensorStorage::Cuda`). Created via `Tensor::new` (zeroed) or
  `Tensor::init_from_cpu` (uploads initial data); read back via `to_cpu()`,
  overwritten via `copy_from_cpu()`.
- **`ShaderDef`** — a named operation with a fixed binding layout
  (`TensorMode::Input/Output/InOut/Meta` per binding slot), implemented once
  as WGSL (for the Vulkan path) and once as CUDA C / a cuBLAS call (for the
  CUDA path). See `BuiltInShader` in `src/nn/shaders.rs` for the full catalog
  (MatMul, MatMulTrp, MatMulWeightBwd, RMSNorm + backward, Softmax + backward,
  RoPE + backward, SiLU + backward, embedding gather/scatter + backward,
  cross-entropy + backward, AdamW, and the multi-head-attention-specific
  HeadGather/HeadScatter/ZeroTensor ops).
- **`ComputeGraph`** — an ordered list of shader dispatches sharing tensor
  bindings. Built once per layer (`add_node` per op), then re-executed every
  training step via `execute()`. On the Vulkan path, `execute()` records one
  command encoder/compute pass per call; on CUDA, it dispatches the matching
  cuBLAS call or NVRTC kernel per node directly.
- **`fuse_compute_graphs`** — concatenates several layers' graphs (e.g. every
  transformer block's forward pass, end to end) into one graph, so a full
  model forward or backward pass is a single `execute()` call instead of
  dozens of small ones.

## Backend implementation notes

### Vulkan/wgpu (`src/context.rs`, `src/graph.rs`)

- **Pipeline caching.** `ComputeGraph::add_node` compiles a shader module +
  bind group layout + pipeline once per *unique shader name* and caches it on
  `WgpuDevice::pipeline_cache`, rather than recompiling for every node. A fused
  model graph reuses the same shader (e.g. `HeadGather`) across every
  attention head in every layer — thousands of nodes — so without this cache,
  model construction created thousands of redundant pipeline objects.
- **Periodic device polling.** `wgpu` only processes its internal
  command-buffer-pool bookkeeping when something polls the device. Training
  calls `execute()` back-to-back, hundreds of times per second, across
  several `ComputeGraph`s sharing one device (forward/backward/AdamW) with no
  other poll point in between. Without polling, wgpu can recycle a command
  buffer that's still pending on the GPU — surfaces as a Vulkan validation
  error (`vkQueueSubmit(): ... is already in use`) and, without validation
  layers active, as an outright `wgpu error: ... Parent device is lost`
  crash. `execute()` now blocks (`Maintain::Wait`) every `POLL_INTERVAL` (2)
  submissions and otherwise just polls non-blockingly, keeping the pool from
  running too far ahead of the GPU without fully serializing every call.
- **Dispatch dimension limit.** Vulkan caps each compute dispatch dimension at
  65535 workgroups. A 1D dispatch over a large tensor (e.g. a 50257x768
  embedding/lm_head table, ~150,772 workgroups of 256 threads) exceeds that —
  a real driver doesn't reliably reject this cleanly, it can manifest as
  device loss instead of a clean validation error. AdamW's shader (the one
  place that dispatches over a *whole* parameter tensor flattened to 1D) now
  spreads across a 2D grid (`groups_x`/`groups_y`, both `<workgroups>` in
  size) instead.

### CUDA (`src/context/cuda.rs`, `src/graph/cuda_exec.rs`)

- **Kernel caching.** `CudaContext::get_or_compile` NVRTC-compiles each named
  kernel once and caches the resulting `CudaFunction` by name.
- **Single stream.** cuBLAS is initialized against the *same* `CudaStream`
  used for NVRTC kernel launches and all `Tensor` htod/dtoh transfers, so
  every dispatch on a given model is strictly ordered without needing
  explicit cross-stream synchronization.
- **Meta-tensor caching.** Meta tensors (`TensorMode::Meta`, e.g. shape
  configs) are read once at graph-construction time and cached as raw bytes
  (`CudaNodeBinding::cached_meta`), since they're normally write-once. The one
  exception is AdamW's `StepConfig` (carries the live `step`/`lr`, mutated
  every training step), which is always re-read live via `live_meta_bytes`
  instead.

## A real bug worth knowing about (now fixed)

`Linear`'s backward `grad_input` computation (in `akasha-core`, but it
exercises wilupgu's `MatMulTrp` shader) used to pass the *forward* pass's
`[M, N, K]` meta tensor into the backward `grad_input` dispatch. `MatMulTrp`
computes `C[M,N] = A[M,K] @ B^T` where `B` must be stored `[N,K]` — but
`grad_input`'s actual contraction needs `N`/`K` swapped relative to the
forward meta. Both backends "faithfully" executed the mislabeled dispatch,
but via different mechanics (cuBLAS's column-major reinterpretation vs WGSL's
direct strided indexing), so they silently produced *different* wrong
results rather than consistently-wrong-but-matching ones. This corrupted the
gradient flowing back from the output head into the rest of the network on
*every* training step, on both backends, and was the actual root cause of a
training run that never converged properly. Confirmed fixed via a tiny
single-layer memorization test (loss → 0.0000 on both backends from
identical seeded initial weights) — see `akasha-core/src/bin/diagnose.rs`
CHECK 8.

## License

No license file is currently present in this repository.
