//! Dispatches a CUDA-backend `ComputeGraph`'s nodes. Elementwise / structured
//! ops are launched as NVRTC-compiled kernels (cached on `CudaContext`);
//! MatMul/MatMulTrp/MatMulWeightBwd go through cuBLAS SGEMM instead.
//!
//! cuBLAS is column-major; our tensors are row-major. To compute row-major
//! C[m,n] = A[m,k] x B[k,n] we call cuBLAS sgemm(NoTrans, NoTrans, n, m, k,
//! alpha, B, n, A, k, beta, C, n) — i.e. treat the row-major matrices as
//! their column-major transposes and swap the operand order, which yields
//! C^T = B^T x A^T in column-major, the bit-identical layout to row-major C.

use super::{ComputeNode, CudaNodeBinding};
use crate::context::CudaContext;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::Gemm;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Meta tensors are always stored as raw bytes inside an f32-typed
/// `CudaSlice<f32>` buffer (mirroring `Tensor::to_cpu`'s use of
/// `bytemuck::cast_slice` to reinterpret storage regardless of its original
/// element type). These two helpers fetch the meta tensor's bytes,
/// reinterpreted as the requested element type, exactly like
/// `Tensor::to_cpu::<u32>()` / `to_cpu::<f32>()` would on the wgpu backend.
///
/// Write-once meta tensors (the vast majority — seq_len, dim, head_dim,
/// M/N/K, scale, eps, etc.) were already read back to host ONCE at
/// `add_node()`/graph-construction time (see `graph.rs`) and cached in
/// `CudaNodeBinding::cached_meta`, so this is just a cheap clone of already
/// host-resident bytes — no blocking GPU readback on the hot dispatch path.
///
/// The sole exception is AdamW's `cfg` (`StepConfig`) binding, which is
/// mutated via `copy_from_cpu` every training step; `launch_adamw` below
/// reads that one directly with `live_meta_bytes` instead of calling this
/// function, so it always sees the fresh value.
fn meta_bytes(b: &CudaNodeBinding, _ctx: &CudaContext) -> Vec<u8> {
    b.cached_meta
        .clone()
        .expect("meta_bytes called on a binding with no cached meta readback (not TensorMode::Meta?)")
}

/// Always-fresh variant of `meta_bytes`: does a blocking `clone_dtoh` every
/// call. Needed only for meta tensors that are mutated after graph
/// construction (currently just AdamW's `cfg`/`StepConfig` binding).
fn live_meta_bytes(b: &CudaNodeBinding, ctx: &CudaContext) -> Vec<u8> {
    let guard = b.slice.lock().unwrap();
    let f32_data = ctx.stream.clone_dtoh(&*guard).expect("meta dtoh failed");
    let byte_len = b.size as usize;
    let bytes: &[u8] = bytemuck::cast_slice(&f32_data);
    bytes[..byte_len.min(bytes.len())].to_vec()
}

/// `cudarc` 0.19's kernel launch is a builder (`stream.launch_builder(&f)`)
/// rather than the older tuple-args `f.launch(cfg, (args...))` API. This
/// thin helper keeps the call sites below close to that older, more
/// readable shape.
macro_rules! launch {
    ($ctx:expr, $f:expr, $cfg:expr, $($arg:expr),+ $(,)?) => {{
        let mut builder = $ctx.stream.launch_builder(&$f);
        $(builder.arg($arg);)+
        unsafe { builder.launch($cfg) }.expect("[wilupgu] CUDA kernel launch failed")
    }};
}

fn meta_u32(b: &CudaNodeBinding, ctx: &CudaContext) -> Vec<u32> {
    bytemuck::cast_slice(&meta_bytes(b, ctx)).to_vec()
}

fn find<'a>(bindings: &'a [CudaNodeBinding], idx: u32) -> &'a CudaNodeBinding {
    bindings
        .iter()
        .find(|b| b.binding == idx)
        .expect("missing binding index")
}

pub fn execute_cuda_graph(ctx: &Arc<CudaContext>, nodes: &[ComputeNode]) {
    for node in nodes {
        let ComputeNode::Cuda {
            name,
            bindings,
            workgroups,
        } = node
        else {
            continue;
        };

        match name.as_str() {
            "MatMul" => gemm_matmul(ctx, bindings, false),
            "MatMulTrp" => gemm_matmul(ctx, bindings, true),
            "MatMulWeightBwd" => gemm_weight_bwd(ctx, bindings),

            "Embedding" => launch_embedding(ctx, bindings, "embedding", crate::nn::cuda_kernels::EMBEDDING, "embedding_kernel", *workgroups),
            "EmbeddingBwd" => launch_embedding(ctx, bindings, "embedding_bwd", crate::nn::cuda_kernels::EMBEDDING_BWD, "embedding_bwd_kernel", *workgroups),

            "CausalMask" => launch_causal_mask(ctx, bindings, *workgroups),
            "HeadGather" => launch_head_move(ctx, bindings, "head_gather", crate::nn::cuda_kernels::HEAD_GATHER, "head_gather_kernel"),
            "HeadScatter" => launch_head_move(ctx, bindings, "head_scatter", crate::nn::cuda_kernels::HEAD_SCATTER, "head_scatter_kernel"),
            "ZeroTensor" => launch_zero_tensor(ctx, bindings),
            "SiLU" => launch_elementwise_1(ctx, bindings, "silu", crate::nn::cuda_kernels::SILU, "silu_kernel"),
            "SiLUBwd" => launch_elementwise_3(ctx, bindings, "silu_bwd", crate::nn::cuda_kernels::SILU_BWD, "silu_bwd_kernel"),
            "ResidualAdd" => launch_add(ctx, bindings, "add", crate::nn::cuda_kernels::ADD, "add_kernel"),
            "BwdAddInplace" => launch_add(ctx, bindings, "bwd_add_inplace", crate::nn::cuda_kernels::BWD_ADD_INPLACE, "bwd_add_inplace_kernel"),

            "RoPE" => launch_rope(ctx, bindings, "rope", crate::nn::cuda_kernels::ROPE, "rope_kernel", *workgroups),
            "RoPEBwd" => launch_rope(ctx, bindings, "rope_bwd", crate::nn::cuda_kernels::ROPE_BWD, "rope_bwd_kernel", *workgroups),

            "Softmax" => launch_softmax(ctx, bindings, "softmax", crate::nn::cuda_kernels::SOFTMAX, "softmax_kernel"),
            "SoftmaxBwd" => launch_softmax_bwd(ctx, bindings),

            "RMSNorm" => launch_rmsnorm(ctx, bindings),
            "RMSNormBwd" => launch_rmsnorm_bwd(ctx, bindings),
            "RMSNormWeightBwd" => launch_rmsnorm_weight_bwd(ctx, bindings),

            "CrossEntropy" => launch_cross_entropy(ctx, bindings),
            "CrossEntropyBwd" => launch_cross_entropy_bwd(ctx, bindings),

            "AdamW" => launch_adamw(ctx, bindings),

            other => panic!("[wilupgu] CUDA backend: no kernel mapping for shader '{other}'"),
        }
    }
}

// ---------------------------------------------------------------- MatMul ---

/// MatMul / MatMulTrp: C[M,N] = A[M,K] x op(B). For MatMul, op(B) = B[K,N].
/// For MatMulTrp, op(B) = B^T where B is stored as [N,K] (see matmul_trp.wgsl:
/// `B[col * config.K + b_col]`, i.e. B's rows are indexed by `col` — the
/// *output* column — confirming B is [N,K] and we need B^T[K,N]).
fn gemm_matmul(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding], transpose_b: bool) {
    let a = find(bindings, 0);
    let b = find(bindings, 1);
    let c = find(bindings, 2);
    let meta = find(bindings, 3);

    let dims = meta_u32(meta, ctx);
    let (m, n, k) = (dims[0], dims[1], dims[2]);

    let a_guard = a.slice.lock().unwrap();
    let b_guard = b.slice.lock().unwrap();
    let mut c_guard = c.slice.lock().unwrap();

    // Row-major C[m,n] = A[m,k] * B[k,n]  <=>  column-major C^T[n,m] = B^T * A^T.
    // cuBLAS sees our row-major buffers as already-transposed column-major
    // matrices, so: gemm(opB, opA, n, m, k, alpha, B, ldb, A, lda, beta, C, n).
    // For MatMulTrp, B is logically [N,K] and we want B^T[K,N], i.e. in cuBLAS's
    // column-major view B is already [K,N]-as-stored == needs `Trans` to act
    // as [N,K]; concretely: opB = Trans, ldb = K.
    let (op_b, ldb) = if transpose_b {
        (cublasOperation_t::CUBLAS_OP_T, k)
    } else {
        (cublasOperation_t::CUBLAS_OP_N, n)
    };

    let cfg = cudarc::cublas::GemmConfig {
        transa: op_b,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: ldb as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };

    unsafe {
        ctx.blas
            .gemm(cfg, &*b_guard, &*a_guard, &mut *c_guard)
            .expect("[wilupgu] cuBLAS sgemm failed (MatMul)");
    }
}

/// MatMulWeightBwd: dB[K,N] += A^T[K,M] x dC[M,N], where A is [M,K] (see
/// matmul_weight_trp.wgsl: `A[m_idx_a * config.K + row]` — row indexes K,
/// so A^T is being formed from A[M,K]).
fn gemm_weight_bwd(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let a = find(bindings, 0); // [M, K]
    let dc = find(bindings, 1); // [M, N]
    let db = find(bindings, 2); // [K, N], accumulated in place
    let meta = find(bindings, 3);

    let dims = meta_u32(meta, ctx);
    let (m, n, k) = (dims[0], dims[1], dims[2]);

    let a_guard = a.slice.lock().unwrap();
    let dc_guard = dc.slice.lock().unwrap();
    let mut db_guard = db.slice.lock().unwrap();

    // Row-major dB[K,N] += A^T[K,M] * dC[M,N]
    // <=> column-major dB^T[N,K] = dC^T[N,M] * A[M,K] (+ accumulate)
    // cuBLAS: gemm(opA=N on dC-as-stored[N,M]... ) — A is row-major [M,K], so
    // its column-major-as-stored view is [K,M]; we want A (not A^T) in the
    // column-major product, so opA = Trans on the stored-as-[K,M] view gives
    // logical [M,K]... Concretely or just match dims with cuBLAS's own
    // row-major trick:
    //   gemm(opDc=Trans, opA=Trans, n=N, m=K, k=M, dC(ld=N), A(ld=K), dB(ld=N), beta=1)
    let cfg = cudarc::cublas::GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_T,
        m: n as i32,
        n: k as i32,
        k: m as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 1.0f32,
        ldc: n as i32,
    };

    unsafe {
        ctx.blas
            .gemm(cfg, &*dc_guard, &*a_guard, &mut *db_guard)
            .expect("[wilupgu] cuBLAS sgemm failed (MatMulWeightBwd)");
    }
}

// ----------------------------------------------------------- Elementwise ---

fn cfg_1d(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn elem_count(b: &CudaNodeBinding) -> u32 {
    (b.size / 4) as u32
}

fn launch_elementwise_1(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding], key: &str, src: &str, func: &str) {
    let x = find(bindings, 0);
    let n = elem_count(x);
    let f = ctx.get_or_compile(key, src, func);
    let mut guard = x.slice.lock().unwrap();
    launch!(ctx, f, cfg_1d(n), &mut *guard, &n);
}

fn launch_elementwise_3(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding], key: &str, src: &str, func: &str) {
    let x = find(bindings, 0);
    let dy = find(bindings, 1);
    let dx = find(bindings, 2);
    let n = elem_count(x);
    let f = ctx.get_or_compile(key, src, func);
    let xg = x.slice.lock().unwrap();
    let dyg = dy.slice.lock().unwrap();
    let mut dxg = dx.slice.lock().unwrap();
    launch!(ctx, f, cfg_1d(n), &*xg, &*dyg, &mut *dxg, &n);
}

fn launch_add(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding], key: &str, src: &str, func: &str) {
    let x = find(bindings, 0);
    let r = find(bindings, 1);
    let n = elem_count(x);
    let f = ctx.get_or_compile(key, src, func);
    let mut xg = x.slice.lock().unwrap();
    let rg = r.slice.lock().unwrap();
    launch!(ctx, f, cfg_1d(n), &mut *xg, &*rg, &n);
}

fn launch_embedding(
    ctx: &Arc<CudaContext>,
    bindings: &[CudaNodeBinding],
    key: &str,
    src: &str,
    func: &str,
    workgroups: [u32; 3],
) {
    // Forward: bindings are tokens(u32), weight(f32), output(f32), meta.
    // Backward: tokens(u32), grad_output(f32), grad_table(f32), meta.
    let b0 = find(bindings, 0);
    let b1 = find(bindings, 1);
    let b2 = find(bindings, 2);
    let meta = find(bindings, 3);
    let dims = meta_u32(meta, ctx);
    let (vocab_size, embed_dim, seq_len) = (dims[0], dims[1], dims[2]);

    let f = ctx.get_or_compile(key, src, func);
    let g0 = b0.slice.lock().unwrap();
    let g1 = b1.slice.lock().unwrap();
    let mut g2 = b2.slice.lock().unwrap();

    let cfg = LaunchConfig {
        grid_dim: (workgroups[0].max(1), seq_len.max(1), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    launch!(ctx, f, cfg, &*g0, &*g1, &mut *g2, &vocab_size, &embed_dim, &seq_len);
}

fn launch_causal_mask(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding], _workgroups: [u32; 3]) {
    let scores = find(bindings, 0);
    let meta = find(bindings, 1);
    // Meta { seq_len: u32, scale: f32 }
    let bytes = meta_bytes(meta, ctx);
    let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
    let scale = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());

    let f = ctx.get_or_compile("causal_mask", crate::nn::cuda_kernels::CAUSAL_MASK, "causal_mask_kernel");
    let mut g = scores.slice.lock().unwrap();

    let grid = (seq_len + 15) / 16;
    let cfg = LaunchConfig {
        grid_dim: (grid, grid, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    launch!(ctx, f, cfg, &mut *g, &seq_len, &scale);
}

fn launch_head_move(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding], key: &str, src: &str, func: &str) {
    let from = find(bindings, 0);
    let to = find(bindings, 1);
    let meta = find(bindings, 2);
    // Meta { seq_len: u32, full_dim: u32, head_dim: u32, head_offset: u32 }
    let dims = meta_u32(meta, ctx);
    let (seq_len, full_dim, head_dim, head_offset) = (dims[0], dims[1], dims[2], dims[3]);

    let f = ctx.get_or_compile(key, src, func);
    let from_g = from.slice.lock().unwrap();
    let mut to_g = to.slice.lock().unwrap();

    let grid = (head_dim + 15) / 16;
    let grid_y = (seq_len + 15) / 16;
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), grid_y.max(1), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    launch!(ctx, f, cfg, &*from_g, &mut *to_g, &seq_len, &full_dim, &head_dim, &head_offset);
}

fn launch_zero_tensor(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let x = find(bindings, 0);
    let n = elem_count(x);
    let f = ctx.get_or_compile("zero_tensor", crate::nn::cuda_kernels::ZERO_TENSOR, "zero_tensor_kernel");
    let mut g = x.slice.lock().unwrap();
    launch!(ctx, f, cfg_1d(n), &mut *g, &n);
}

fn launch_rope(
    ctx: &Arc<CudaContext>,
    bindings: &[CudaNodeBinding],
    key: &str,
    src: &str,
    func: &str,
    _workgroups: [u32; 3],
) {
    let vec_b = find(bindings, 0);
    let meta = find(bindings, 1);
    let dims = meta_u32(meta, ctx);
    let (seq_len, dim, head_dim) = (dims[0], dims[1], dims[2]);

    let f = ctx.get_or_compile(key, src, func);
    let mut g = vec_b.slice.lock().unwrap();

    let grid_x = (head_dim / 2 + 15) / 16;
    let grid_y = (seq_len + 15) / 16;
    let cfg = LaunchConfig {
        grid_dim: (grid_x.max(1), grid_y.max(1), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    launch!(ctx, f, cfg, &mut *g, &seq_len, &dim, &head_dim);
}

fn launch_softmax(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding], key: &str, src: &str, func: &str) {
    let x = find(bindings, 0);
    let meta = find(bindings, 1);
    let seq_len = meta_u32(meta, ctx)[0];

    let f = ctx.get_or_compile(key, src, func);
    let mut g = x.slice.lock().unwrap();
    launch!(ctx, f, cfg_1d(seq_len), &mut *g, &seq_len);
}

fn launch_softmax_bwd(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let y = find(bindings, 0);
    let dy = find(bindings, 1);
    let dx = find(bindings, 2);
    let meta = find(bindings, 3);
    // Meta { seq_len: u32, scale: f32 }
    let bytes = meta_bytes(meta, ctx);
    let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
    let scale = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());

    let f = ctx.get_or_compile(
        "softmax_bwd",
        crate::nn::cuda_kernels::SOFTMAX_BWD,
        "softmax_bwd_kernel",
    );
    let yg = y.slice.lock().unwrap();
    let dyg = dy.slice.lock().unwrap();
    let mut dxg = dx.slice.lock().unwrap();
    launch!(ctx, f, cfg_1d(seq_len), &*yg, &*dyg, &mut *dxg, &seq_len, &scale);
}

fn launch_rmsnorm(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let x = find(bindings, 0);
    let w = find(bindings, 1);
    let out = find(bindings, 2);
    let meta = find(bindings, 3);
    // Meta { seq_len: u32, size: u32, eps: f32 }
    let bytes = meta_bytes(meta, ctx);
    let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
    let size = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
    let eps = f32::from_ne_bytes(bytes[8..12].try_into().unwrap());

    let f = ctx.get_or_compile("rmsnorm", crate::nn::cuda_kernels::RMSNORM, "rmsnorm_kernel");
    let xg = x.slice.lock().unwrap();
    let wg = w.slice.lock().unwrap();
    let mut og = out.slice.lock().unwrap();

    let cfg = LaunchConfig {
        grid_dim: (seq_len.max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    launch!(ctx, f, cfg, &*xg, &*wg, &mut *og, &seq_len, &size, &eps);
}

fn launch_rmsnorm_bwd(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let dy = find(bindings, 0);
    let x = find(bindings, 1);
    let w = find(bindings, 2);
    let dx = find(bindings, 3);
    let rsqrt_cache = find(bindings, 4);
    let meta = find(bindings, 5);
    // Meta { seq_len: u32, size: u32, eps: f32 }
    let bytes = meta_bytes(meta, ctx);
    let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
    let size = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
    let eps = f32::from_ne_bytes(bytes[8..12].try_into().unwrap());

    let f = ctx.get_or_compile(
        "rmsnorm_bwd",
        crate::nn::cuda_kernels::RMSNORM_BWD,
        "rmsnorm_bwd_kernel",
    );
    let dyg = dy.slice.lock().unwrap();
    let xg = x.slice.lock().unwrap();
    let wg = w.slice.lock().unwrap();
    let mut dxg = dx.slice.lock().unwrap();
    let mut rsg = rsqrt_cache.slice.lock().unwrap();

    let cfg = LaunchConfig {
        grid_dim: (seq_len.max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    launch!(
        ctx, f, cfg,
        &*dyg, &*xg, &*wg, &mut *dxg, &mut *rsg, &seq_len, &size, &eps
    );
}

fn launch_rmsnorm_weight_bwd(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let dy = find(bindings, 0);
    let x = find(bindings, 1);
    let rsqrt_cache = find(bindings, 2);
    let dweight = find(bindings, 3);
    let meta = find(bindings, 4);
    let dims = meta_u32(meta, ctx);
    let (seq_len, size) = (dims[0], dims[1]);

    let f = ctx.get_or_compile(
        "rmsnorm_weight_bwd",
        crate::nn::cuda_kernels::RMSNORM_WEIGHT_BWD,
        "rmsnorm_weight_bwd_kernel",
    );
    let dyg = dy.slice.lock().unwrap();
    let xg = x.slice.lock().unwrap();
    let rsg = rsqrt_cache.slice.lock().unwrap();
    let mut dwg = dweight.slice.lock().unwrap();

    launch!(ctx, f, cfg_1d(size), &*dyg, &*xg, &*rsg, &mut *dwg, &seq_len, &size);
}

fn launch_cross_entropy(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let logits = find(bindings, 0);
    let targets = find(bindings, 1);
    let probs = find(bindings, 2);
    let losses = find(bindings, 3);
    let meta = find(bindings, 4);
    let dims = meta_u32(meta, ctx);
    let (vocab_size, num_rows) = (dims[0], dims[1]);

    let f = ctx.get_or_compile(
        "cross_entropy",
        crate::nn::cuda_kernels::CROSS_ENTROPY,
        "cross_entropy_kernel",
    );
    let lg = logits.slice.lock().unwrap();
    let tg = targets.slice.lock().unwrap();
    let mut pg = probs.slice.lock().unwrap();
    let mut losg = losses.slice.lock().unwrap();

    launch!(ctx, f, cfg_1d(num_rows), &*lg, &*tg, &mut *pg, &mut *losg, &vocab_size, &num_rows);
}

fn launch_cross_entropy_bwd(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let probs = find(bindings, 0);
    let targets = find(bindings, 1);
    let d_losses = find(bindings, 2);
    let d_logits = find(bindings, 3);
    let meta = find(bindings, 4);
    let dims = meta_u32(meta, ctx);
    let (vocab_size, num_rows) = (dims[0], dims[1]);

    let f = ctx.get_or_compile(
        "cross_entropy_bwd",
        crate::nn::cuda_kernels::CROSS_ENTROPY_BWD,
        "cross_entropy_bwd_kernel",
    );
    let pg = probs.slice.lock().unwrap();
    let tg = targets.slice.lock().unwrap();
    let dlg = d_losses.slice.lock().unwrap();
    let mut dlogg = d_logits.slice.lock().unwrap();

    launch!(ctx, f, cfg_1d(num_rows), &*pg, &*tg, &*dlg, &mut *dlogg, &vocab_size, &num_rows);
}

fn launch_adamw(ctx: &Arc<CudaContext>, bindings: &[CudaNodeBinding]) {
    let weights = find(bindings, 0);
    let grads = find(bindings, 1);
    let m = find(bindings, 2);
    let v = find(bindings, 3);
    let param_meta = find(bindings, 4);
    let cfg_meta = find(bindings, 5);

    let size = meta_u32(param_meta, ctx)[0];
    // StepConfig layout (see adamw.wgsl): { step: u32, lr: f32, beta1: f32,
    // beta2: f32, eps: f32, weight_decay: f32 } — read as raw bytes and
    // reinterpret each field by its real type rather than assuming all-f32.
    //
    // IMPORTANT: unlike every other meta tensor read in this file, `cfg_meta`
    // (binding 5 of the "AdamW" shader) is mutated every single training
    // step via `AdamW::step()` -> `self.cfg.copy_from_cpu(&[StepConfig{..}])`
    // in akasha-core/src/optim/adamw.rs, so it must NOT use the cached
    // graph-construction-time snapshot — always do a live GPU readback here.
    let bytes = live_meta_bytes(cfg_meta, ctx);
    let step = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
    let lr = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());
    let beta1 = f32::from_ne_bytes(bytes[8..12].try_into().unwrap());
    let beta2 = f32::from_ne_bytes(bytes[12..16].try_into().unwrap());
    let eps = f32::from_ne_bytes(bytes[16..20].try_into().unwrap());
    let weight_decay = f32::from_ne_bytes(bytes[20..24].try_into().unwrap());

    let f = ctx.get_or_compile("adamw", crate::nn::cuda_kernels::ADAMW, "adamw_kernel");
    let mut wg = weights.slice.lock().unwrap();
    let gg = grads.slice.lock().unwrap();
    let mut mg = m.slice.lock().unwrap();
    let mut vg = v.slice.lock().unwrap();

    launch!(
        ctx, f, cfg_1d(size),
        &mut *wg, &*gg, &mut *mg, &mut *vg,
        &size, &step, &lr, &beta1, &beta2, &eps, &weight_decay
    );
}
