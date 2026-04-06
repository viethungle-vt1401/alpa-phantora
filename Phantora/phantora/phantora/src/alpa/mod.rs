use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use good_lp::*;

// --- 1. Data Structures ---
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ComputeNode {
    pub name: String,
    pub op: String,
    pub target: String,
    pub args: Vec<String>,
    pub output_shape: Option<Vec<i64>>,
}

pub type ComputeGraph = Vec<ComputeNode>;

// --- 2. Topological Sort ---
pub fn topological_sort(graph: &ComputeGraph) -> Vec<ComputeNode> {
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();
    let mut node_map: HashMap<String, ComputeNode> = HashMap::new();

    for node in graph {
        in_degree.insert(node.name.clone(), 0);
        node_map.insert(node.name.clone(), node.clone());
        adj_list.insert(node.name.clone(), Vec::new());
    }

    for node in graph {
        for arg in &node.args {
            if let Some(neighbors) = adj_list.get_mut(arg) {
                neighbors.push(node.name.clone());
                *in_degree.entry(node.name.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut queue: VecDeque<String> = VecDeque::new();
    for (name, &deg) in &in_degree {
        if deg == 0 {
            queue.push_back(name.clone());
        }
    }

    let mut sorted_sequence = Vec::new();
    while let Some(name) = queue.pop_front() {
        let node = node_map.get(&name).unwrap().clone();
        
        if node.op != "placeholder" && node.op != "output" && node.op != "get_attr" {
            sorted_sequence.push(node);
        }

        if let Some(neighbors) = adj_list.get(&name) {
            for neighbor in neighbors {
                let deg = in_degree.get_mut(neighbor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(neighbor.clone());
                }
            }
        }
    }

    sorted_sequence
}

// --- 3. Intra-Op ILP Solver ---
const DEVICE_MEMORY_LIMIT_BYTES: f64 = 16.0 * 1024.0 * 1024.0 * 1024.0; // Back to 16GB
const ALPHA_COMM: f64 = 0.5;
const BETA_COMM: f64 = 0.05;
const RESHARDING_PENALTY: f64 = 2.5;

#[derive(Clone, Copy)]
pub struct IntraStrategy {
    pub compute_time: f64,
    pub memory_used: f64,
    pub valid: bool,
}

pub fn solve_intra_ilp(stage_layers: &[ComputeNode], devices: usize) -> Option<IntraStrategy> {
    if stage_layers.is_empty() || devices == 0 { return None; }

    let mut problem = ProblemVariables::new();
    let num_layers = stage_layers.len();

    let valid_strategies: Vec<usize> = vec![1, 2, 4, 8]
        .into_iter()
        .filter(|&p| p <= devices && devices % p == 0)
        .collect();
    let num_strats = valid_strategies.len();

    let mut x_vars: Vec<Vec<Variable>> = vec![];
    for _ in 0..num_layers {
        let mut layer_vars = vec![];
        for _ in 0..num_strats {
            layer_vars.push(problem.add(variable().binary()));
        }
        x_vars.push(layer_vars);
    }

    let mut y_vars: Vec<Vec<Vec<Variable>>> = vec![];
    for _ in 1..num_layers {
        let mut layer_edges = vec![];
        for _ in 0..num_strats {
            let mut edge_to = vec![];
            for _ in 0..num_strats {
                edge_to.push(problem.add(variable().binary()));
            }
            layer_edges.push(edge_to);
        }
        y_vars.push(layer_edges);
    }

    let mut objective = Expression::from(0.0);
    let mut total_memory_expr = Expression::from(0.0);

    for (i, node) in stage_layers.iter().enumerate() {
        let elements = node.output_shape.as_ref().unwrap_or(&vec![1]).iter().product::<i64>() as f64;
        let base_compute = elements / 1000.0;
        let base_memory = elements * 4.0;

        for (s_idx, &p) in valid_strategies.iter().enumerate() {
            let mem_per_device = base_memory / (p as f64);
            let compute_cost = (base_compute / p as f64) + ALPHA_COMM + (BETA_COMM * mem_per_device);
            
            objective += x_vars[i][s_idx] * compute_cost;
            total_memory_expr += x_vars[i][s_idx] * mem_per_device;
        }
    }

    for i in 1..num_layers {
        for (s1_idx, &p1) in valid_strategies.iter().enumerate() {
            for (s2_idx, &p2) in valid_strategies.iter().enumerate() {
                if p1 != p2 {
                    objective += y_vars[i-1][s1_idx][s2_idx] * RESHARDING_PENALTY;
                }
            }
        }
    }

    let mut model = problem.minimise(objective.clone()).using(highs);
    for i in 0..num_layers {
        let mut one_strat = Expression::from(0.0);
        for s_idx in 0..num_strats {
            one_strat += x_vars[i][s_idx];
        }
        model = model.with(constraint!(one_strat == 1.0));
    }

    model = model.with(constraint!(total_memory_expr <= DEVICE_MEMORY_LIMIT_BYTES));

    for i in 1..num_layers {
        for s1_idx in 0..num_strats {
            for s2_idx in 0..num_strats {
                let y = y_vars[i-1][s1_idx][s2_idx];
                let x_prev = x_vars[i-1][s1_idx];
                let x_curr = x_vars[i][s2_idx];
                model = model.with(constraint!(y >= x_prev + x_curr - 1.0));
            }
        }
    }

    if let Ok(solution) = model.solve() {
        let optimal_time = solution.eval(objective);
        let mut final_memory = 0.0;
        for (i, node) in stage_layers.iter().enumerate() {
            let elements = node.output_shape.as_ref().unwrap_or(&vec![1]).iter().product::<i64>() as f64;
            for (s_idx, &p) in valid_strategies.iter().enumerate() {
                if solution.value(x_vars[i][s_idx]) > 0.5 { 
                    final_memory += (elements * 4.0) / (p as f64);
                }
            }
        }

        Some(IntraStrategy {
            compute_time: optimal_time,
            memory_used: final_memory,
            valid: true,
        })
    } else {
        None
    }
}

// --- 4. Inter-Op DP Search ---
fn build_ilp_cost_matrix(layers: &[ComputeNode], max_devices: usize) -> Vec<Vec<Vec<Option<IntraStrategy>>>> {
    let n = layers.len();
    let mut matrix = vec![vec![vec![None; max_devices + 1]; n]; n];

    for start in 0..n {
        for end in start..n {
            let stage_layers = &layers[start..=end];
            for d in 1..=max_devices {
                matrix[start][end][d] = solve_intra_ilp(stage_layers, d);
            }
        }
    }
    matrix
}

fn dp_inner_ilp_pruned(
    layers: &[ComputeNode],
    ilp_matrix: &Vec<Vec<Vec<Option<IntraStrategy>>>>,
    num_devices: usize,
    t_max_limit: f64,
) -> Option<(f64, Vec<Vec<(usize, usize)>>)> {
    let num_layers = layers.len();
    let mut dp = vec![vec![f64::INFINITY; num_devices + 1]; num_layers + 1];
    let mut cuts = vec![vec![(0, 0); num_devices + 1]; num_layers + 1];
    
    dp[0][0] = 0.0;

    for i in 1..=num_layers {
        for d in 1..=num_devices {
            for j in 0..i {
                for k in 0..d {
                    let devices_for_stage = d - k;
                    if let Some(strat) = &ilp_matrix[j][i - 1][devices_for_stage] {
                        if strat.memory_used <= DEVICE_MEMORY_LIMIT_BYTES && strat.compute_time <= t_max_limit {
                            let total_cost = dp[j][k] + strat.compute_time;
                            if total_cost < dp[i][d] {
                                dp[i][d] = total_cost;
                                cuts[i][d] = (j, k);
                            }
                        }
                    }
                }
            }
        }
    }

    if dp[num_layers][num_devices] == f64::INFINITY { None } else { Some((dp[num_layers][num_devices], cuts)) }
}

pub fn find_optimal_pipeline(layers: &[ComputeNode], num_devices: usize, num_microbatches: f64) -> (f64, Vec<(usize, usize, usize)>) {
    println!("Precomputing exact ILP intra-op strategies...");
    let ilp_matrix = build_ilp_cost_matrix(layers, num_devices);
    
    // FIX: Calculate a robust 'high' bound dynamically.
    // 999,999.0 was too small for massive LLM graphs!
    let mut max_single_layer_cost = 0.0;
    for i in 0..layers.len() {
        // Look at the base cost of putting this single layer on 1 device
        if let Some(strat) = &ilp_matrix[i][i][1] {
            max_single_layer_cost += strat.compute_time;
        }
    }
    
    let mut low = 0.0;
    // Safe upper bound: Worst case all layers computed sequentially with massive headroom
    let mut high = if max_single_layer_cost > 0.0 { max_single_layer_cost * 10.0 } else { 1e12 };
    
    println!("DP Binary Search Bounds -> Low: {:.2}, High: {:.2}", low, high);
    
    let mut best_pipeline_latency = f64::INFINITY;
    let mut best_cuts = None;

    for _ in 0..50 { 
        let current_t_max = (low + high) / 2.0;

        if let Some((min_sum_latencies, cuts)) = dp_inner_ilp_pruned(layers, &ilp_matrix, num_devices, current_t_max) {
            let pipeline_latency = (num_microbatches - 1.0) * current_t_max + min_sum_latencies;
            if pipeline_latency < best_pipeline_latency {
                best_pipeline_latency = pipeline_latency;
                best_cuts = Some(cuts); 
            }
            high = current_t_max; 
        } else {
            low = current_t_max; 
        }
    }
    
    if best_cuts.is_none() {
        println!("CRITICAL ERROR: DP Failed to find a valid pipeline. The model exceeds the {} GB memory limit even when fully sharded.", DEVICE_MEMORY_LIMIT_BYTES / 1e9);
    }

    let mut schedule = Vec::new();
    if let Some(cuts) = best_cuts {
        let mut curr_layer = layers.len();
        let mut curr_dev = num_devices;
        while curr_layer > 0 && curr_dev > 0 {
            let (prev_layer, prev_dev) = cuts[curr_layer][curr_dev];
            schedule.push((prev_layer, curr_layer - 1, curr_dev - prev_dev));
            curr_layer = prev_layer;
            curr_dev = prev_dev;
        }
        schedule.reverse();
    }
    (best_pipeline_latency, schedule)
}

// --- 5. Simulator Integration Bridge ---
use crate::torch_call::TorchCall;

#[derive(Clone)]
pub struct ExecutionPlan {
    pub stages: Vec<Stage>,
}

#[derive(Clone)]
pub struct Stage {
    pub ops: Vec<usize>,
    pub devices: Vec<usize>,
}

pub fn build_graph(calls: &[TorchCall]) -> ComputeGraph {
    // Map the runtime TorchCalls to our standard ComputeNode structure.
    // Creates sequential dependencies to ensure the DP formulation runs smoothly.
    calls
        .iter()
        .enumerate()
        .map(|(i, _c)| ComputeNode {
            name: format!("call_{}", i),
            op: "simulated_torch_op".to_string(),
            target: "".to_string(),
            args: if i > 0 { vec![format!("call_{}", i - 1)] } else { vec![] },
            output_shape: Some(vec![1, 1024, 1024]), // default fallback shape
        })
        .collect()
}

/// The entry point called by `simulator.rs`
pub fn plan(calls: &[TorchCall], num_devices: usize) -> ExecutionPlan {
    // 1. Convert the simulator's runtime calls into our ComputeGraph format
    let graph = build_graph(calls);

    // 2. Run the topological sort required by our new DP solver
    let sorted_layers = topological_sort(&graph);

    // 3. Run the exact ILP + DP search (Using a default microbatch count of 64.0 for simulation)
    let (_optimal_time, schedule) = find_optimal_pipeline(&sorted_layers, num_devices, 64.0);

    // 4. Map the new search schedule back into the ExecutionPlan format expected by Phantora
    let mut stages = Vec::new();
    for (start, end, devs) in schedule {
        stages.push(Stage {
            ops: (start..=end).collect(),
            devices: (0..devs).collect(),
        });
    }

    ExecutionPlan { stages }
}