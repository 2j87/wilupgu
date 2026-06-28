//! Standalone correctness check for the meta-tensor caching fix in
//! `src/graph.rs` / `src/graph/cuda_exec.rs`.
//!
//! Verifies two things on the CUDA backend:
//!   1. A small `ComputeGraph` (RMSNorm) executed TWICE in a row produces
//!      identical, correct output both times -- i.e. caching the meta
//!      tensor's bytes at `add_node()` time didn't break repeated dispatch.
//!   2. A minimal AdamW-style node, re-stepped 3 times with a DIFFERENT
//!      `lr` each time (by mutating the `cfg` meta tensor via
//!      `copy_from_cpu` between executes, exactly like the real
//!      `AdamW::step()` does), actually uses the fresh `lr` each time --
//!      i.e. the AdamW `cfg` binding is correctly exempted from caching and
//!      still does a live GPU readback every dispatch.
//!
//! Run with: cargo run --release --features cuda --example meta_cache_check

#[cfg(feature = "cuda")]
fn main() {
    use std::sync::Arc;
    use wilupgu::context::WgpuContext;
    use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
    use wilupgu::nn::shaders::BuiltInShader;
    use wilupgu::tensor::Tensor;

    let ctx = Arc::new(
        WgpuContext::new_cuda().expect("CUDA backend unavailable -- this check requires a CUDA GPU"),
    );

    // ---------------------------------------------------------------
    // Check 1: RMSNorm executed twice produces identical, correct output.
    // ---------------------------------------------------------------
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct RmsMeta {
        seq_len: u32,
        size: u32,
        eps: f32,
    }

    let seq_len: u32 = 2;
    let size: u32 = 4;
    let x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.0];
    let w_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    let x = Arc::new(Tensor::init_from_cpu(ctx.clone(), &x_data));
    let w = Arc::new(Tensor::init_from_cpu(ctx.clone(), &w_data));
    let out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0f32; 8]));
    let meta = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &[RmsMeta {
            seq_len,
            size,
            eps: 1e-5,
        }],
    ));

    let mut graph = ComputeGraph::new(ctx.clone());
    let def = BuiltInShader::RMSNorm.get_def();
    graph.add_node(
        &def,
        &[
            TensorBind { binding: 0, tensor: &x, mode: TensorMode::Input },
            TensorBind { binding: 1, tensor: &w, mode: TensorMode::Input },
            TensorBind { binding: 2, tensor: &out, mode: TensorMode::Output },
            TensorBind { binding: 3, tensor: &meta, mode: TensorMode::Meta },
        ],
        [seq_len, 1, 1],
    );

    graph.execute();
    let run1: Vec<f32> = out.to_cpu();

    graph.execute();
    let run2: Vec<f32> = out.to_cpu();

    assert_eq!(run1, run2, "RMSNorm output differed between two consecutive execute() calls!");

    // Expected RMSNorm: y = x / sqrt(mean(x^2) + eps) * w, per row of `size` elems.
    let expected: Vec<f32> = x_data
        .chunks(size as usize)
        .flat_map(|row| {
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / size as f32;
            let scale = 1.0 / (mean_sq + 1e-5).sqrt();
            row.iter().map(move |v| v * scale).collect::<Vec<_>>()
        })
        .collect();

    for (a, b) in run1.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-3, "RMSNorm result {a} differs from expected {b}");
    }

    println!("[check 1] RMSNorm: two consecutive execute() calls -> identical, correct output. OK");
    println!("  run1 = {:?}", run1);
    println!("  run2 = {:?}", run2);

    // ---------------------------------------------------------------
    // Check 2: AdamW-style node, 3 steps with different `lr`/`step` each
    // time, confirms cfg_meta (binding 5) is read live, not cached.
    // ---------------------------------------------------------------
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ParamMeta {
        size: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct StepConfig {
        step: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    }

    let n: u32 = 4;
    let weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1.0f32, 1.0, 1.0, 1.0]));
    let grad = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1.0f32, 1.0, 1.0, 1.0]));
    let m = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[0.0f32; 4]));
    let v = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[0.0f32; 4]));
    let param_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[ParamMeta { size: n }]));
    let cfg = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &[StepConfig {
            step: 0,
            lr: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }],
    ));

    let mut adamw_graph = ComputeGraph::new(ctx.clone());
    let adamw_def = BuiltInShader::AdamW.get_def();
    adamw_graph.add_node(
        &adamw_def,
        &[
            TensorBind { binding: 0, tensor: &weight, mode: TensorMode::InOut },
            TensorBind { binding: 1, tensor: &grad, mode: TensorMode::Input },
            TensorBind { binding: 2, tensor: &m, mode: TensorMode::InOut },
            TensorBind { binding: 3, tensor: &v, mode: TensorMode::InOut },
            TensorBind { binding: 4, tensor: &param_meta, mode: TensorMode::Meta },
            TensorBind { binding: 5, tensor: &cfg, mode: TensorMode::Meta },
        ],
        [(n + 255) / 256, 1, 1],
    );

    let lrs = [0.1f32, 0.01, 0.001];
    let mut weight_snapshots = Vec::new();
    for (i, &lr) in lrs.iter().enumerate() {
        cfg.copy_from_cpu(&[StepConfig {
            step: (i + 1) as u32,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }]);
        adamw_graph.execute();
        weight_snapshots.push(weight.to_cpu::<f32>());
    }

    println!("[check 2] AdamW weight after each step with lr = {:?}:", lrs);
    for (lr, w) in lrs.iter().zip(weight_snapshots.iter()) {
        println!("  lr={lr} -> weight={:?}", w);
    }

    // With weight_decay=0, AdamW's first-step update magnitude is
    // approximately `lr` (since m_hat/sqrt(v_hat+eps) ~= sign(grad) = 1 on
    // step 1, bias-corrected). The successive per-step deltas should shrink
    // roughly in proportion to the shrinking `lr` -- if the cfg meta tensor
    // were stale/cached, every step would apply the SAME (first) lr and the
    // deltas would stay constant instead of shrinking by ~10x each step.
    let delta = |a: &Vec<f32>, b: &Vec<f32>| (a[0] - b[0]).abs();
    let d1 = (1.0f32 - weight_snapshots[0][0]).abs();
    let d2 = delta(&weight_snapshots[0], &weight_snapshots[1]);
    let d3 = delta(&weight_snapshots[1], &weight_snapshots[2]);

    println!("  per-step deltas: d1={d1:.6} (lr=0.1), d2={d2:.6} (lr=0.01), d3={d3:.6} (lr=0.001)");

    assert!(d1 > d2 * 3.0, "step1->2 delta did not shrink as expected when lr dropped 10x (cfg may be stale/cached)");
    assert!(d2 > d3 * 3.0, "step2->3 delta did not shrink as expected when lr dropped 10x (cfg may be stale/cached)");

    println!("[check 2] AdamW: per-step deltas shrink in line with falling lr -> cfg_meta is read LIVE each dispatch, not cached. OK");

    println!("\nAll checks passed.");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires --features cuda. Run: cargo run --release --features cuda --example meta_cache_check");
}
