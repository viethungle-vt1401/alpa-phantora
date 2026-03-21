use phantora::args::get_args;
use phantora::simulator;
use phantora::torch_call::{TorchCall, TorchCallInfo, TensorInfo};

use cuda_call::{capi, CudaCallMsg, CudaStream, HostId, ResponseId};

use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::net::UnixDatagram;

fn main() {
    let env = env_logger::Env::new()
        .filter("PHANTORA_LOG")
        .write_style("PHANTORA_LOG_STYLE");
    env_logger::init_from_env(env);

    let _args = get_args();
    main_loop();
}

#[derive(Serialize, Deserialize)]
enum Message {
    Cuda(CudaCallMsg),
    Torch(phantora::torch_call::TorchCallMsg, i32),
    Exit(ResponseId, i64),
}

fn main_loop() {
    // Setup simulator (same as before)
    let socket_path = capi::simulator_socket_path();
    let _ = fs::remove_file(&socket_path);
    let recv_socket = UnixDatagram::bind(socket_path).unwrap();

    let mut buf = [0u8; 1024 * 1024];

    let netconfig: netsim::config::Config =
        netsim::config::read_config(&get_args().net_config);
    println!("netconfig: {:#?}", netconfig);

    let cluster = netsim::config::build_cloud(&netconfig);
    let netsim = netsim::simulator::SimulatorBuilder::new()
        .with_setting(netconfig.simulator)
        .cluster(cluster)
        .host_mapping(netconfig.host_mapping)
        .build()
        .expect("Fail to create network simulator");

    let mut simulator = simulator::Simulator::new(netsim);

    // 🔥 SYNTHETIC WORKLOAD (THIS IS THE KEY PART)
    println!("INJECTING SYNTHETIC WORKLOAD");

    for i in 0..50 {
        let call = TorchCall {
            time: i * 10,
            id: ResponseId {
                host: HostId {
                    hostname: "host-1".to_string(),
                    pid: 0,
                },
                tid: 0,
            },
            stream: CudaStream { device: 0, id: 0 },
            info: TorchCallInfo::MM(
                TensorInfo {
                    shape: vec![1024, 1024],
                    dtype: tch::Kind::Float,
                },
                TensorInfo {
                    shape: vec![1024, 1024],
                    dtype: tch::Kind::Float,
                },
            ),
        };

        simulator.handle_torch_call(call);
    }

    // 🔥 Stop here for synthetic mode
    println!("SYNTHETIC RUN COMPLETE");
    return;

    // --- ORIGINAL LOOP (kept for completeness, unreachable in synthetic mode) ---
    #[allow(unreachable_code)]
    loop {
        let sz = recv_socket.recv(&mut buf).unwrap();
        assert!(sz < buf.len());
        let last = sz - 1;
        let tag = buf[last];
        let buf = &buf[..last];

        let message = match tag {
            1 => {
                let msg = bincode::deserialize::<CudaCallMsg>(&buf).unwrap();
                Message::Cuda(msg)
            }
            2 => {
                let info = std::str::from_utf8(&buf).unwrap();
                let callmsg =
                    serde_json::from_str::<phantora::torch_call::TorchCallJson>(info)
                        .unwrap()
                        .into_msg();

                if let Some(arg_device) = callmsg.gpu_index() {
                    Message::Torch(callmsg, arg_device as _)
                } else {
                    continue;
                }
            }
            3 => panic!("Exit handling omitted in synthetic mode"),
            _ => panic!("Unknown message tag {}", tag),
        };

        match message {
            Message::Cuda(msg) => simulator.handle_cuda_call(msg),
            Message::Torch(callmsg, arg_device) => {
                if let Some(call) = callmsg.into_call(arg_device) {
                    simulator.handle_torch_call(call);
                }
            }
            Message::Exit(_, _) => {}
        }
    }
}