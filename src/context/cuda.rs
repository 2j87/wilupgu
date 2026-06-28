use cudarc::cublas::CudaBlas;
use cudarc::driver::result::DriverError;
use cudarc::driver::{CudaContext as CuDevice, CudaFunction, CudaStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// CUDA device + cuBLAS handle + a lazily-populated cache of NVRTC-compiled
/// kernels (keyed by kernel name). Mirrors the role `WgpuDevice` plays for the
/// wgpu backend: it's the thing every `Tensor`/`ComputeGraph` on this backend
/// is built against.
///
/// Named `CudaContext` (distinct from `cudarc::driver::CudaContext`, aliased
/// here as `CuDevice`) to match the naming this task asked for.
pub struct CudaContext {
    pub device: Arc<CuDevice>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub(crate) kernel_cache: Mutex<HashMap<String, CudaFunction>>,
}

impl CudaContext {
    pub fn new(ordinal: usize) -> Result<Self, DriverError> {
        let device = CuDevice::new(ordinal)?;
        let stream = device.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| {
            // cublas::result::CublasError doesn't convert to DriverError; wrap as a
            // generic driver error string via panic-free fallback.
            eprintln!("[wilupgu] failed to initialize cuBLAS: {e:?}");
            DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)
        })?;

        Ok(Self {
            device,
            stream,
            blas,
            kernel_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Compiles (if not already cached) and returns the named kernel function
    /// from the given CUDA C source. `func_name` must match a `extern "C"
    /// __global__` symbol inside `src`.
    pub(crate) fn get_or_compile(
        &self,
        cache_key: &str,
        src: &str,
        func_name: &str,
    ) -> CudaFunction {
        {
            let cache = self.kernel_cache.lock().unwrap();
            if let Some(f) = cache.get(cache_key) {
                return f.clone();
            }
        }

        let ptx = cudarc::nvrtc::compile_ptx(src)
            .unwrap_or_else(|e| panic!("[wilupgu] NVRTC compile failed for '{cache_key}': {e:?}"));

        let module = self
            .device
            .load_module(ptx)
            .unwrap_or_else(|e| panic!("[wilupgu] failed to load PTX module '{cache_key}': {e:?}"));

        let func = module
            .load_function(func_name)
            .unwrap_or_else(|e| panic!("[wilupgu] kernel '{func_name}' not found in module '{cache_key}': {e:?}"));

        self.kernel_cache
            .lock()
            .unwrap()
            .insert(cache_key.to_string(), func.clone());

        func
    }
}
