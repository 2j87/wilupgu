use std::sync::Arc;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::CudaContext;

/// Pure-wgpu device/queue pair. Kept as its own struct (rather than folded into
/// `WgpuContext`) so the wgpu/Vulkan path is byte-for-byte what it was before
/// the CUDA backend was added.
pub struct WgpuDevice {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    /// Compiled (bind group layout, pipeline) cache keyed by shader name.
    /// Without this, `ComputeGraph::add_node` previously compiled a brand new
    /// shader module + bind group layout + pipeline layout + pipeline for
    /// *every* node, even when the same shader (e.g. HeadGather, used by every
    /// attention head in every layer) was reused thousands of times across a
    /// fused model graph. That's thousands of redundant pipeline objects kept
    /// alive simultaneously, dispatched within a single compute pass at
    /// `execute()` time -- a very plausible source of the Vulkan "Parent
    /// device is lost" crashes seen during full-scale training (which never
    /// happened on the CUDA backend, which already caches compiled kernels by
    /// name via `CudaContext::kernel_cache`). This mirrors that same pattern.
    pub(crate) pipeline_cache:
        std::sync::Mutex<std::collections::HashMap<String, (Arc<wgpu::BindGroupLayout>, Arc<wgpu::ComputePipeline>)>>,
    /// Counts `ComputeGraph::execute()` calls across *all* graphs sharing this
    /// device (forward/backward/AdamW each have their own `ComputeGraph`, but
    /// only one underlying queue). Submitting many command buffers back-to-back
    /// with zero polling lets wgpu's internal command-buffer pool race ahead of
    /// GPU completion (manifests as "VkCommandBuffer ... before it has
    /// completed" / "Parent device is lost"); blocking on every single
    /// submission fixes that but serializes CPU/GPU and tanks throughput. This
    /// counter drives a periodic blocking wait instead (see `execute()`).
    pub(crate) submit_count: std::sync::atomic::AtomicU64,
}

impl WgpuDevice {
    async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Wilupgu_Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipeline_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            submit_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

/// Backend-agnostic context. This is the type referenced throughout wilupgu
/// and akasha-core as `WgpuContext` for historical/back-compat reasons; it now
/// wraps either a wgpu (Vulkan) device or, when built with `--features cuda`,
/// a CUDA device + cuBLAS handle. All existing call sites that construct it
/// via `WgpuContext::new().await` and pass it around as `Arc<WgpuContext>`
/// keep working unmodified — the default constructor always resolves to the
/// `Wgpu` variant, exactly matching pre-existing behavior.
pub enum WgpuContext {
    Wgpu(WgpuDevice),
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaContext>),
}

impl WgpuContext {
    /// Default constructor: always selects the wgpu/Vulkan backend, matching
    /// pre-existing behavior exactly. Use `new_cuda()` to opt into CUDA.
    pub async fn new() -> Self {
        WgpuContext::Wgpu(WgpuDevice::new().await)
    }

    #[cfg(feature = "cuda")]
    pub fn new_cuda() -> Option<Self> {
        match CudaContext::new(0) {
            Ok(ctx) => {
                println!("[wilupgu] CUDA backend selected (NVIDIA GPU detected)");
                Some(WgpuContext::Cuda(Arc::new(ctx)))
            }
            Err(e) => {
                eprintln!(
                    "[wilupgu] CUDA backend unavailable ({e:?}), falling back to wgpu/Vulkan"
                );
                None
            }
        }
    }

    pub fn is_cuda(&self) -> bool {
        match self {
            WgpuContext::Wgpu(_) => false,
            #[cfg(feature = "cuda")]
            WgpuContext::Cuda(_) => true,
        }
    }

    /// Panics if this context isn't a wgpu context. Internal helper used by
    /// code paths that are inherently wgpu-only (e.g. building wgpu pipelines).
    pub(crate) fn wgpu(&self) -> &WgpuDevice {
        match self {
            WgpuContext::Wgpu(d) => d,
            #[cfg(feature = "cuda")]
            WgpuContext::Cuda(_) => panic!("[wilupgu] expected a wgpu context, got CUDA"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda(&self) -> &Arc<CudaContext> {
        match self {
            WgpuContext::Cuda(c) => c,
            WgpuContext::Wgpu(_) => panic!("[wilupgu] expected a CUDA context, got wgpu"),
        }
    }
}
