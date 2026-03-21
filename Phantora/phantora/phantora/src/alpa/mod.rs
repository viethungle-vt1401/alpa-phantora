use crate::torch_call::TorchCall;

pub mod graph;
pub mod intra;
pub mod inter;
pub mod mesh;

use graph::*;
use intra::*;
use inter::*;
use mesh::*;

pub fn plan(calls: &[TorchCall], num_devices: usize) -> ExecutionPlan {
    let graph = build_graph(calls);

    // 1. Intra-op strategy search
    let intra = solve_intra(&graph, num_devices);

    // 2. Inter-op DP partitioning
    let stages = solve_inter(&graph, &intra, num_devices);

    ExecutionPlan { stages }
}

#[derive(Clone)]
pub struct ExecutionPlan {
    pub stages: Vec<Stage>,
}

#[derive(Clone)]
pub struct Stage {
    pub ops: Vec<usize>,
    pub devices: Vec<usize>,
}