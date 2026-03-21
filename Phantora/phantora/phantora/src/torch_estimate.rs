// use crate::cuda_bindings::*;
use crate::torch_call::{TensorInfo, TorchCall, TorchCallInfo};
// use crate::{assert_cuda, estimate};
use lru::LruCache;
use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::NonZeroUsize;
// use std::ptr;
use std::time::Duration;
use tch::{self, Device, Kind, Tensor};

enum KindRange {
    Integer,
    Bool,
    Float,
}

fn kind_range(kind: Kind) -> KindRange {
    match kind {
        Kind::Bool => KindRange::Bool,
        Kind::Uint8
        | Kind::Int8
        | Kind::Int16
        | Kind::UInt16
        | Kind::Int
        | Kind::Int64
        | Kind::UInt32
        | Kind::UInt64 => KindRange::Integer,
        _ => KindRange::Float,
    }
}

// macro_rules! estimate_torch {
//     ($n:expr, $e:expr) => {{
//         tch::autocast(true, || estimate!($n, $e))
//     }};
// }

pub struct TorchEstimator {
    tensor_cache: LruCache<(i64, Kind), Tensor>,
    compute_cache: HashMap<TorchCallInfo, Duration>,
    sequence_cache: BTreeMap<u64, Vec<Duration>>,
    device: Device,
}

impl TorchEstimator {
    pub fn new() -> Self {
        let device = Device::Cpu;

        // warmup (CPU)
        let mat = Tensor::randn(&[1024, 1024], (Kind::Float, device));
        let _ = mat.mm(&mat);

        println!("TorchEstimator initialized (CPU mode)");

        Self {
            tensor_cache: LruCache::new(NonZeroUsize::new(32).unwrap()),
            compute_cache: HashMap::new(),
            sequence_cache: BTreeMap::new(),
            device,
        }
    }

    fn allocate(&mut self, info: &TensorInfo) -> Tensor {
        let shape = info.shape.as_slice();
        let kind = info.dtype;
        let total_size = shape.iter().product();

        match self.tensor_cache.get(&(total_size, kind)) {
            Some(t) => t.view_(shape),
            None => {
                let t = match kind_range(kind) {
                    KindRange::Bool => Tensor::randint(2, shape, (kind, self.device)),
                    KindRange::Integer => Tensor::randint(128, shape, (kind, self.device)),
                    KindRange::Float => Tensor::randn(shape, (kind, self.device)),
                };
                self.tensor_cache.put((total_size, kind), t.view_(shape));
                t
            }
        }
    }

    // fn allocate_list(&mut self, info: &[TensorInfo]) -> Vec<Tensor> {
    //     info.iter().map(|info| self.allocate(info)).collect()
    // }

    fn cache(&mut self, t: Tensor) {
        if let Device::Cpu = t.device() {
            let total_size = t.size().iter().product();
            let kind = t.kind();
            self.tensor_cache.put((total_size, kind), t);
        }
    }

    fn run(&mut self, _niter: i32, call: &TorchCallInfo) -> Duration {
        match call {
            TorchCallInfo::MM(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let start = std::time::Instant::now();
                let result = t1.mm(&t2);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::MatMul(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let start = std::time::Instant::now();
                let result = t1.matmul(&t2);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::Linear(info1, info2, bias_info) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let bias = bias_info.as_ref().map(|info| self.allocate(info));
                let start = std::time::Instant::now();
                let result = t1.linear::<&Tensor>(&t2, bias.as_ref());
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::BMM(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let start = std::time::Instant::now();
                let result = t1.bmm(&t2);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::AddMM(info1, info2, info3) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let start = std::time::Instant::now();
                let result = t1.addmm(&t2, &t3);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::BAddBMM(info1, info2, info3) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let start = std::time::Instant::now();
                let result = t1.baddbmm(&t2, &t3, 1, 1);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::Mul(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let start = std::time::Instant::now();
                let result = t1.g_mul(&t2);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::MulScalar(info) => {
                let t = self.allocate(info);
                let start = std::time::Instant::now();
                let result = t.multiply_scalar(2);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::Add(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let start = std::time::Instant::now();
                let result = t1.g_add(&t2);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::Div(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let start = std::time::Instant::now();
                let result = t1.g_div(&t2);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::Softmax(info, dim) => {
                let t = self.allocate(info);
                let start = std::time::Instant::now();
                let result = t.softmax(*dim, info.dtype);
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::ZerosLike(info) => {
                let t = self.allocate(info);
                let start = std::time::Instant::now();
                let result = t.zeros_like();
                let dur = start.elapsed();
                self.cache(result);
                dur
            }

            TorchCallInfo::SDPA { q, k, v, causal, gqa } => {
                let t_q = self.allocate(q);
                let t_k = self.allocate(k);
                let t_v = self.allocate(v);

                let start = std::time::Instant::now();
                let result = Tensor::f_scaled_dot_product_attention(
                    &t_q,
                    &t_k,
                    &t_v,
                    None::<Tensor>,
                    0.0,
                    *causal,
                    None,
                    *gqa
                ).unwrap();
                let dur = start.elapsed();

                self.cache(result);
                dur
            }

            _ => Duration::from_micros(10),
        }
    }

    pub fn estimate(&mut self, call: &TorchCallInfo) -> Duration {
        if let Some(value) = self.compute_cache.get(call) {
            *value
        } else {
            let duration = self.run(2, call);
            self.compute_cache.insert(call.clone(), duration);
            duration
        }
    }

    pub fn estimate_sequence(&mut self, calls: &[TorchCall]) -> Vec<Duration> {
        let seq_hash = {
            let mut hasher = DefaultHasher::new();
            for call in calls {
                TorchCallInfo::hash(&call.info, &mut hasher);
            }
            hasher.finish()
        };

        if let Some(value) = self.sequence_cache.get(&seq_hash) {
            return value.clone();
        }

        let mut durs = Vec::new();
        for call in calls {
            durs.push(self.run(1, &call.info));
        }

        self.sequence_cache.insert(seq_hash, durs.clone());
        durs
    }
}
