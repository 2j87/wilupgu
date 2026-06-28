use crate::context::WgpuContext;
use crate::tensor::Tensor;
use std::sync::Arc;

mod fuse;
pub use fuse::fuse_compute_graphs;

#[cfg(feature = "cuda")]
mod cuda_exec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorMode {
    Input,  // (Read-Only)
    Output, // (Write-Only)
    InOut,  // (Read-Write)
    Meta,   // (Read-Only)
}

pub struct ShaderDef {
    pub name: String,
    pub source: String,
    pub expected_layout: Vec<TensorMode>,
}

impl ShaderDef {
    pub fn new(name: &str, source: &str, expected_layout: Vec<TensorMode>) -> Self {
        Self {
            name: name.to_string(),
            source: source.to_string(),
            expected_layout,
        }
    }
}

pub struct TensorBind<'a> {
    pub binding: u32,
    pub tensor: &'a Tensor,
    pub mode: TensorMode,
}

/// A single binding captured for a CUDA-backed node: the CUDA slice handle
/// (shared via `Arc<Mutex<..>>`, cloned from the originating `Tensor`), its
/// binding index and `TensorMode`, plus the tensor's byte size (used to
/// recover element counts for meta-tensor readback at dispatch time).
#[cfg(feature = "cuda")]
pub struct CudaNodeBinding {
    pub binding: u32,
    pub slice: Arc<std::sync::Mutex<cudarc::driver::CudaSlice<f32>>>,
    pub mode: TensorMode,
    pub size: u64,
    /// One-time host-side readback of this binding's bytes, captured at
    /// `add_node()` time (graph-construction time, not the hot dispatch
    /// loop). Populated only for `TensorMode::Meta` bindings whose contents
    /// are write-once (the overwhelming majority of meta tensors: seq_len,
    /// dim, head_dim, M/N/K, scale, eps, etc. — set via `Tensor::init_from_cpu`
    /// and never mutated again). `None` for Input/Output/InOut bindings.
    ///
    /// NOTE: a handful of meta tensors ARE mutated post-construction via
    /// `Tensor::copy_from_cpu` (currently only AdamW's `cfg`/`StepConfig`
    /// binding, which carries a fresh `step`/`lr` every training step).
    /// `cuda_exec.rs` special-cases that one binding (shader "AdamW",
    /// binding index 5) to always re-read live from the GPU instead of using
    /// this cache, regardless of what's stored here.
    pub cached_meta: Option<Vec<u8>>,
}

/// A single dispatch in a `ComputeGraph`. The wgpu variant holds a compiled
/// pipeline + bind group exactly as before. The CUDA variant instead keeps
/// the bound tensors plus enough metadata (shader name, workgroup counts) for
/// `cuda_exec` to dispatch the matching cuBLAS call or NVRTC kernel at
/// `execute()` time.
pub enum ComputeNode {
    Wgpu {
        name: String,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: Arc<wgpu::BindGroup>,
        workgroups: [u32; 3],
    },
    #[cfg(feature = "cuda")]
    Cuda {
        name: String,
        bindings: Vec<CudaNodeBinding>,
        workgroups: [u32; 3],
    },
}

pub struct ComputeGraph {
    ctx: Arc<WgpuContext>,
    nodes: Vec<ComputeNode>,
}

impl ComputeGraph {
    pub fn new(ctx: Arc<WgpuContext>) -> Self {
        Self {
            ctx,
            nodes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, shader: &ShaderDef, bindings: &[TensorBind], workgroups: [u32; 3]) {
        // ---------------- VALIDATION ------------------
        if bindings.len() != shader.expected_layout.len() {
            panic!(
                "[ERROR] Shader Binding Mismatch: Shader '{}' expects {} tensors, but {} were provided.",
                shader.name, shader.expected_layout.len(), bindings.len()
            );
        }

        for bind in bindings {
            let expected_mode = shader.expected_layout[bind.binding as usize];
            if bind.mode != expected_mode {
                panic!(
                    "[ERROR] Tensor Mode Mismatch: In shader '{}', binding {} expects {:?}, but {:?} was provided.",
                    shader.name, bind.binding, expected_mode, bind.mode
                );
            }
        }
        // ---------------------------------------------------

        #[cfg(feature = "cuda")]
        if self.ctx.is_cuda() {
            let cuda_ctx = self.ctx.cuda();
            let cuda_bindings = bindings
                .iter()
                .map(|b| match &b.tensor.storage {
                    crate::tensor::TensorStorage::Cuda(s) => {
                        // Meta tensors are write-once (set via
                        // `Tensor::init_from_cpu` and never mutated again,
                        // with the sole exception of AdamW's `cfg` binding
                        // which `cuda_exec.rs` is responsible for always
                        // re-reading live). Do the blocking dtoh readback
                        // here, once, at graph-construction time, instead of
                        // on every `execute()` dispatch.
                        let cached_meta = if b.mode == TensorMode::Meta {
                            let guard = s.lock().unwrap();
                            let f32_data = cuda_ctx
                                .stream
                                .clone_dtoh(&*guard)
                                .expect("meta dtoh failed");
                            let byte_len = b.tensor.size as usize;
                            let bytes: &[u8] = bytemuck::cast_slice(&f32_data);
                            Some(bytes[..byte_len.min(bytes.len())].to_vec())
                        } else {
                            None
                        };

                        CudaNodeBinding {
                            binding: b.binding,
                            slice: s.clone(),
                            mode: b.mode,
                            size: b.tensor.size,
                            cached_meta,
                        }
                    }
                    crate::tensor::TensorStorage::Wgpu(_) => {
                        panic!("[wilupgu] CUDA graph received a wgpu-backed tensor")
                    }
                })
                .collect();

            self.nodes.push(ComputeNode::Cuda {
                name: shader.name.clone(),
                bindings: cuda_bindings,
                workgroups,
            });
            return;
        }

        let wgpu_ctx = self.ctx.wgpu();

        // Compiled (layout, pipeline) is purely a function of the shader --
        // identical for every node using the same shader regardless of which
        // tensors are bound -- so cache it by name instead of recompiling a
        // fresh shader module + pipeline for every single node.
        let (bind_group_layout, pipeline) = {
            let mut cache = wgpu_ctx.pipeline_cache.lock().unwrap();
            if let Some((layout, pipeline)) = cache.get(&shader.name) {
                (layout.clone(), pipeline.clone())
            } else {
                let shader_module = wgpu_ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&format!("{}_Shader", shader.name)),
                        source: wgpu::ShaderSource::Wgsl(shader.source.as_str().into()),
                    });

                let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = shader
                    .expected_layout
                    .iter()
                    .enumerate()
                    .map(|(i, mode)| {
                        let read_only = match mode {
                            TensorMode::Input | TensorMode::Meta => true,
                            TensorMode::Output | TensorMode::InOut => false,
                        };
                        wgpu::BindGroupLayoutEntry {
                            binding: i as u32,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }
                    })
                    .collect();

                let bind_group_layout =
                    wgpu_ctx
                        .device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some(&format!("{}_Layout", shader.name)),
                            entries: &layout_entries,
                        });

                let pipeline_layout =
                    wgpu_ctx
                        .device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some(&format!("{}_PipelineLayout", shader.name)),
                            bind_group_layouts: &[&bind_group_layout],
                            push_constant_ranges: &[],
                        });

                let pipeline = wgpu_ctx
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(&format!("{}_Pipeline", shader.name)),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: "main",
                    });

                let layout = Arc::new(bind_group_layout);
                let pipeline = Arc::new(pipeline);
                cache.insert(shader.name.clone(), (layout.clone(), pipeline.clone()));
                (layout, pipeline)
            }
        };

        let bind_entries: Vec<wgpu::BindGroupEntry> = bindings
            .iter()
            .map(|bind| {
                let buffer = match &bind.tensor.storage {
                    crate::tensor::TensorStorage::Wgpu(buf) => buf,
                    #[cfg(feature = "cuda")]
                    crate::tensor::TensorStorage::Cuda(_) => {
                        panic!("[wilupgu] wgpu graph received a CUDA-backed tensor")
                    }
                };
                wgpu::BindGroupEntry {
                    binding: bind.binding,
                    resource: buffer.as_entire_binding(),
                }
            })
            .collect();

        let bind_group = wgpu_ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{}_BindGroup", shader.name)),
                layout: &bind_group_layout,
                entries: &bind_entries,
            });

        self.nodes.push(ComputeNode::Wgpu {
            name: shader.name.clone(),
            pipeline,
            bind_group: Arc::new(bind_group),
            workgroups,
        });
    }

    pub fn execute(&self) {
        #[cfg(feature = "cuda")]
        if self.ctx.is_cuda() {
            cuda_exec::execute_cuda_graph(self.ctx.cuda(), &self.nodes);
            return;
        }

        let wgpu_ctx = self.ctx.wgpu();

        let mut encoder = wgpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Wilupgu_Execute_Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Wilupgu_Compute_Pass"),
                timestamp_writes: None,
            });

            for node in &self.nodes {
                // `ComputeNode::Cuda` only exists under `--features cuda`, in which
                // case this whole function isn't reached (we return early above);
                // allow the resulting irrefutable pattern on the non-cuda build.
                #[allow(irrefutable_let_patterns)]
                if let ComputeNode::Wgpu {
                    pipeline,
                    bind_group,
                    workgroups,
                    ..
                } = node
                {
                    cpass.set_pipeline(pipeline);
                    cpass.set_bind_group(0, bind_group, &[]);
                    cpass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                }
            }
        }

        wgpu_ctx.queue.submit(Some(encoder.finish()));

        // wgpu only processes its internal completion bookkeeping (which
        // drives command-buffer-pool recycling) when something polls the
        // device. Training calls execute() back-to-back, hundreds of times
        // per second, across several ComputeGraphs sharing this same device
        // (forward/backward/AdamW), with no other poll point in between
        // (Tensor::to_cpu() does poll, but that's only hit every LOG_EVERY
        // steps) -- without this, wgpu can reuse/recycle a command buffer
        // that's still pending on the GPU, which surfaces as a Vulkan
        // validation error ("on active VkCommandBuffer ... before it has
        // completed") and, without validation layers active, as an outright
        // "Parent device is lost" crash.
        //
        // Blocking on every single submission fixes that but fully
        // serializes CPU/GPU work and tanks throughput (~32 steps/min,
        // ~4.5 days for a 200k-step run). Instead, block only every
        // POLL_INTERVAL submissions -- frequent enough that the command
        // buffer pool never gets more than a handful of submissions ahead of
        // GPU completion, infrequent enough to keep most submissions async.
        const POLL_INTERVAL: u64 = 2;
        let n = wgpu_ctx
            .submit_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if n % POLL_INTERVAL == 0 {
            wgpu_ctx.device.poll(wgpu::Maintain::Wait);
        } else {
            wgpu_ctx.device.poll(wgpu::Maintain::Poll);
        }
    }
}
