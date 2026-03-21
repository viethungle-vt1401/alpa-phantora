// --- constants expected by Phantora ---
pub const NCCL_SPLIT_NOCOLOR: i32 = -1;

// --- FIX: do NOT try to read env var ---
pub fn simulator_socket_path() -> std::ffi::OsString {
    "/tmp/phantora.sock".into()
}

pub fn node_socket_path(_pid: u32, _tid: i32) -> std::ffi::OsString {
    "/tmp/phantora_node.sock".into()
}

// --- REQUIRED extern symbols (minimal stubs) ---

#[no_mangle]
pub extern "C" fn cuda_device_synchronize(_device: i32) {}

#[no_mangle]
pub extern "C" fn cuda_stream_synchronize(_device: i32, _stream: i32) {}

#[no_mangle]
pub extern "C" fn cuda_stream_query(_device: i32, _stream: i32) -> i32 {
    1
}

#[no_mangle]
pub extern "C" fn cuda_add_latency(_device: i32, _stream: i32, _latency: i64) {}

#[no_mangle]
pub extern "C" fn cuda_memcpy_async(
    _src: usize,
    _dst: usize,
    _size: usize,
    _kind: i32,
    _device: i32,
    _stream: i32,
) {}

#[no_mangle]
pub extern "C" fn cuda_event_record(_device: i32, _stream: i32, _id: i32) {}

#[no_mangle]
pub extern "C" fn cuda_event_synchronize(
    _device: i32,
    _stream: i32,
    _id: i32,
) -> i64 {
    0
}

#[no_mangle]
pub extern "C" fn cuda_event_query(
    _device: i32,
    _stream: i32,
    _id: i32,
    _time_ref: *mut i64,
) -> i32 {
    1
}

// --- NCCL stubs ---

#[no_mangle]
pub extern "C" fn nccl_get_unique_id(_id: *mut i8) {}

#[no_mangle]
pub extern "C" fn nccl_group_start() {}

#[no_mangle]
pub extern "C" fn nccl_group_end() {}

#[no_mangle]
pub extern "C" fn nccl_comm_init_rank(
    _nranks: i32,
    _comm_id: *const i8,
    _rank: i32,
    _device: i32,
) {}

#[no_mangle]
pub extern "C" fn nccl_comm_split(
    _rank: i32,
    _comm_id: *const i8,
    _color: i32,
    _key: i32,
    _rank_out: *mut i32,
    _nrank_out: *mut i32,
    _id_out: *mut u8,
) {}

#[no_mangle]
pub extern "C" fn nccl_bcast(
    _count: usize,
    _dtype: i32,
    _root: i32,
    _comm_id: *const i8,
    _rank: i32,
    _device: i32,
    _stream: i32,
) {}

#[no_mangle]
pub extern "C" fn nccl_all_reduce(
    _count: usize,
    _dtype: i32,
    _op: i32,
    _comm_id: *const i8,
    _rank: i32,
    _device: i32,
    _stream: i32,
) {}

#[no_mangle]
pub extern "C" fn nccl_all_gather(
    _count: usize,
    _dtype: i32,
    _comm_id: *const i8,
    _rank: i32,
    _device: i32,
    _stream: i32,
) {}

#[no_mangle]
pub extern "C" fn nccl_reduce_scatter(
    _count: usize,
    _dtype: i32,
    _op: i32,
    _comm_id: *const i8,
    _rank: i32,
    _device: i32,
    _stream: i32,
) {}