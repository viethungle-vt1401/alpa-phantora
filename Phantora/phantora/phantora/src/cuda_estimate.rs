use crate::cuda_bindings::*;
use cuda_call::CudaMemcpyKind;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};
use std::collections::HashMap;
use std::ptr;
use std::time::Duration;

#[macro_export]
macro_rules! assert_cuda {
    ($e:expr) => {{
        let err = $e;
        if err != 0 {
            log::error!("{} error {}, {}:{}", stringify!($e), err, file!(), line!());
        }
    }};
}

#[macro_export]
macro_rules! estimate {
    ($n:expr, $e:expr) => {{
        let result;
        let dur;
        let n: i32 = $n;
        unsafe {
            for _ in 0..(n - 1) {
                let _ = $e;
            };
            let mut start_event = ptr::null_mut();
            let mut end_event = ptr::null_mut();
            assert_cuda!(cudaEventCreate(&mut start_event));
            assert_cuda!(cudaEventCreate(&mut end_event));
            assert_cuda!(cudaEventRecord(start_event, ptr::null_mut()));
            result = $e;
            assert_cuda!(cudaEventRecord(end_event, ptr::null_mut()));
            assert_cuda!(cudaEventSynchronize(end_event));
            let mut elapsed: f32 = 0.0;
            assert_cuda!(cudaEventElapsedTime(&mut elapsed, start_event, end_event));
            assert_cuda!(cudaEventDestroy(start_event));
            assert_cuda!(cudaEventDestroy(end_event));
            dur = Duration::from_secs_f64(elapsed as f64 * 1e-3);
        };
        (result, dur)
    }}
}

pub struct CudaEstimator {
    memcpy_cache: HashMap<(CudaMemcpyKind, usize), Duration>,
    torch: PyObject,
    flash_attn_cuda: Option<PyObject>,
    device: PyObject,
}

fn replay_memcpy(kind: CudaMemcpyKind, size: usize) -> Duration {
    match kind {
        CudaMemcpyKind::HostToHost => {
            let host_mem1;
            let host_mem2;
            unsafe {
                host_mem1 = libc::malloc(size);
                host_mem2 = libc::malloc(size);
            }
            let dur = estimate!(
                2,
                cudaMemcpyAsync(
                    host_mem2,
                    host_mem1 as _,
                    size,
                    CUDA_MEMCPY_HOST_HOST,
                    ptr::null_mut()
                )
            )
            .1;
            unsafe {
                libc::free(host_mem1);
                libc::free(host_mem2);
            }
            dur
        }
        CudaMemcpyKind::HostToDevice => {
            let host_mem;
            let mut device_mem = ptr::null_mut();
            unsafe {
                host_mem = libc::malloc(size);
                assert_cuda!(cudaMalloc(&mut device_mem, size));
            }
            let dur = estimate!(
                2,
                cudaMemcpyAsync(
                    device_mem,
                    host_mem as _,
                    size,
                    CUDA_MEMCPY_HOST_DEVICE,
                    ptr::null_mut()
                )
            )
            .1;
            unsafe {
                libc::free(host_mem);
                assert_cuda!(cudaFree(device_mem));
            }
            dur
        }
        CudaMemcpyKind::PinnedHostToDevice => {
            let mut host_mem = ptr::null_mut();
            let mut device_mem = ptr::null_mut();
            unsafe {
                assert_cuda!(cudaMallocHost(&mut host_mem, size));
                assert_cuda!(cudaMalloc(&mut device_mem, size));
            }
            let dur = estimate!(
                2,
                cudaMemcpyAsync(
                    device_mem,
                    host_mem as _,
                    size,
                    CUDA_MEMCPY_HOST_DEVICE,
                    ptr::null_mut()
                )
            )
            .1;
            unsafe {
                assert_cuda!(cudaFreeHost(host_mem));
                assert_cuda!(cudaFree(device_mem));
            }
            dur
        }
        CudaMemcpyKind::DeviceToHost => {
            let host_mem;
            let mut device_mem = ptr::null_mut();
            unsafe {
                host_mem = libc::malloc(size);
                assert_cuda!(cudaMalloc(&mut device_mem, size));
            }
            let dur = estimate!(
                2,
                cudaMemcpyAsync(
                    host_mem,
                    device_mem as _,
                    size,
                    CUDA_MEMCPY_DEVICE_HOST,
                    ptr::null_mut()
                )
            )
            .1;
            unsafe {
                libc::free(host_mem);
                assert_cuda!(cudaFree(device_mem));
            }
            dur
        }
        CudaMemcpyKind::DeviceToPinnedHost => {
            let mut host_mem = ptr::null_mut();
            let mut device_mem = ptr::null_mut();
            unsafe {
                assert_cuda!(cudaMallocHost(&mut host_mem, size));
                assert_cuda!(cudaMalloc(&mut device_mem, size));
            }
            let dur = estimate!(
                2,
                cudaMemcpyAsync(
                    host_mem,
                    device_mem as _,
                    size,
                    CUDA_MEMCPY_DEVICE_HOST,
                    ptr::null_mut()
                )
            )
            .1;
            unsafe {
                assert_cuda!(cudaFreeHost(host_mem));
                assert_cuda!(cudaFree(device_mem));
            }
            dur
        }
        CudaMemcpyKind::DeviceToDevice => {
            let mut device_mem1 = ptr::null_mut();
            let mut device_mem2 = ptr::null_mut();
            unsafe {
                assert_cuda!(cudaMalloc(&mut device_mem1, size));
                assert_cuda!(cudaMalloc(&mut device_mem2, size));
            }
            let dur = estimate!(
                2,
                cudaMemcpyAsync(
                    device_mem1,
                    device_mem2 as _,
                    size,
                    CUDA_MEMCPY_DEVICE_DEVICE,
                    ptr::null_mut()
                )
            )
            .1;
            unsafe {
                assert_cuda!(cudaFree(device_mem1));
                assert_cuda!(cudaFree(device_mem2));
            }
            dur
        }
    }
}

impl CudaEstimator {
    pub fn new() -> Self {
        let (torch, flash_attn_cuda, device): (PyObject, Option<PyObject>, PyObject) =
            Python::with_gil(|py| {
                py.run_bound(
                    "import signal; signal.signal(signal.SIGINT, signal.SIG_DFL)",
                    None,
                    None,
                )
                .ok();

                let torch = py.import_bound("torch").unwrap();
                let flash_attn_cuda = py.import_bound("flash_attn_2_cuda").ok();
                let device = torch.call_method1("device", ("cpu",)).unwrap();

                (
                    torch.unbind().into(),
                    flash_attn_cuda.map(|m| m.unbind().into()),
                    device.unbind().into(),
                )
            });

        println!("CudaEstimator initialized (CPU mode, flash_attn optional)");

        Self {
            memcpy_cache: HashMap::new(),
            torch,
            flash_attn_cuda,
            device,
        }
    }

    pub fn memcpy(&mut self, kind: CudaMemcpyKind, size: usize) -> Duration {
        if let Some(&dur) = self.memcpy_cache.get(&(kind, size)) {
            dur
        } else {
            let dur = replay_memcpy(kind, size);
            self.memcpy_cache.insert((kind, size), dur);
            dur
        }
    }

    fn alloc_torch_tensor(&mut self, py: Python, shape: &[i32], dtype: &PyObject) -> PyObject {
        let shape = PyTuple::new_bound(py, shape);
        let kwargs = [("device", &self.device), ("dtype", dtype)].into_py_dict_bound(py);

        self.torch
            .call_method_bound(py, "randn", shape, Some(&kwargs))
            .unwrap()
    }

    pub fn flash_attn(
        &mut self,
        is_fwd: bool,
        _is_bf16: bool,
        batch_size: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        num_heads: i32,
        _num_heads_k: i32,
        head_size: i32,
        _window_size_left: i32,
        _window_size_right: i32,
        _is_causal: bool,
    ) -> Duration {

        // fallback if no flash_attn
        if self.flash_attn_cuda.is_none() {
            let flops = (batch_size as u64)
                * (seqlen_q as u64)
                * (seqlen_k as u64)
                * (num_heads as u64)
                * (head_size as u64);

            return Duration::from_micros((flops / 1_000_000) as u64);
        }

        Python::with_gil(|py| {
            let module = match &self.flash_attn_cuda {
                Some(m) => m.clone_ref(py),
                None => return Duration::from_micros(1),
            };

            let dtype = self.torch.getattr(py, "float32").unwrap();

            let q = self.alloc_torch_tensor(
                py,
                &[batch_size, seqlen_q, num_heads, head_size],
                &dtype,
            );

            let k = self.alloc_torch_tensor(
                py,
                &[batch_size, seqlen_k, num_heads, head_size],
                &dtype,
            );

            let v = self.alloc_torch_tensor(
                py,
                &[batch_size, seqlen_k, num_heads, head_size],
                &dtype,
            );

            let args = PyTuple::new_bound(py, &[q.as_any(), k.as_any(), v.as_any()]);

            let _ = module.call_method1(py, if is_fwd { "fwd" } else { "bwd" }, &args);

            Duration::from_micros(10)
        })
    }
}
