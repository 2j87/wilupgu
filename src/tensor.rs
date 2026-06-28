use crate::context::WgpuContext;
use bytemuck::Pod;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

/// Backend-specific storage for a `Tensor`. The wgpu variant is identical to
/// what existed before this refactor (a plain storage buffer); the CUDA
/// variant holds a device slice of f32 behind a Mutex so `Tensor` can stay
/// cheaply clonable via `Arc` the same way the wgpu buffer was.
pub enum TensorStorage {
    Wgpu(Arc<wgpu::Buffer>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<Mutex<CudaSlice<f32>>>),
}

pub struct Tensor {
    pub ctx: Arc<WgpuContext>,
    pub storage: TensorStorage,
    pub size: u64,
}

impl Tensor {
    pub fn new(ctx: Arc<WgpuContext>, size_bytes: u64, usage: wgpu::BufferUsages) -> Self {
        match &*ctx {
            WgpuContext::Wgpu(d) => {
                let buffer = d.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Wilupgu_Tensor"),
                    size: size_bytes,
                    usage,
                    mapped_at_creation: false,
                });
                Self {
                    ctx,
                    storage: TensorStorage::Wgpu(Arc::new(buffer)),
                    size: size_bytes,
                }
            }
            #[cfg(feature = "cuda")]
            WgpuContext::Cuda(c) => {
                let n_elems = (size_bytes as usize) / std::mem::size_of::<f32>();
                let slice = c
                    .stream
                    .alloc_zeros::<f32>(n_elems)
                    .expect("[wilupgu] CUDA alloc failed");
                Self {
                    ctx,
                    storage: TensorStorage::Cuda(Arc::new(Mutex::new(slice))),
                    size: size_bytes,
                }
            }
        }
    }

    pub fn init_from_cpu<T: Pod>(ctx: Arc<WgpuContext>, data: &[T]) -> Self {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        match &*ctx {
            WgpuContext::Wgpu(d) => {
                let buffer = d
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Wilupgu_Tensor"),
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    });
                Self {
                    ctx,
                    storage: TensorStorage::Wgpu(Arc::new(buffer)),
                    size,
                }
            }
            #[cfg(feature = "cuda")]
            WgpuContext::Cuda(c) => {
                let f32_data: &[f32] = bytemuck::cast_slice(data);
                let slice = c
                    .stream
                    .clone_htod(f32_data)
                    .expect("[wilupgu] CUDA htod copy failed");
                Self {
                    ctx,
                    storage: TensorStorage::Cuda(Arc::new(Mutex::new(slice))),
                    size,
                }
            }
        }
    }

    pub fn copy_from_cpu<T: Pod>(&self, data: &[T]) {
        match (&*self.ctx, &self.storage) {
            (WgpuContext::Wgpu(d), TensorStorage::Wgpu(buffer)) => {
                d.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
            }
            #[cfg(feature = "cuda")]
            (WgpuContext::Cuda(c), TensorStorage::Cuda(slice)) => {
                let f32_data: &[f32] = bytemuck::cast_slice(data);
                let mut guard = slice.lock().unwrap();
                c.stream
                    .memcpy_htod(f32_data, &mut *guard)
                    .expect("[wilupgu] CUDA htod copy failed");
            }
            #[cfg(feature = "cuda")]
            _ => unreachable!("[wilupgu] Tensor ctx/storage backend mismatch"),
        }
    }

    pub fn to_cpu<T: Pod + Default + Clone>(&self) -> Vec<T> {
        match (&*self.ctx, &self.storage) {
            (WgpuContext::Wgpu(d), TensorStorage::Wgpu(buffer)) => {
                let device = &d.device;
                let queue = &d.queue;

                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Staging_Buffer"),
                    size: self.size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut encoder = device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, self.size);
                queue.submit(Some(encoder.finish()));

                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

                device.poll(wgpu::Maintain::Wait);
                pollster::block_on(async { receiver.receive().await.unwrap().unwrap() });

                let data = buffer_slice.get_mapped_range();
                let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

                drop(data);
                staging_buffer.unmap();

                result
            }
            #[cfg(feature = "cuda")]
            (WgpuContext::Cuda(c), TensorStorage::Cuda(slice)) => {
                let guard = slice.lock().unwrap();
                let f32_data = c
                    .stream
                    .clone_dtoh(&*guard)
                    .expect("[wilupgu] CUDA dtoh copy failed");
                bytemuck::cast_slice(&f32_data).to_vec()
            }
            #[cfg(feature = "cuda")]
            _ => unreachable!("[wilupgu] Tensor ctx/storage backend mismatch"),
        }
    }

    pub fn free(self) {
        match self.storage {
            TensorStorage::Wgpu(buffer) => buffer.destroy(),
            #[cfg(feature = "cuda")]
            TensorStorage::Cuda(_) => {
                // CudaSlice frees its device memory on Drop; nothing extra to do.
            }
        }
    }
}
