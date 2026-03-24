use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

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
// Flattens the DAG into a strict 1D sequence for the DP solver.
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
            if adj_list.contains_key(arg) {
                adj_list.get_mut(arg).unwrap().push(node.name.clone());
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

// --- 3. Cost Model ---
// Simulates the time (in ms) to compute a single node on a specific number of devices.
fn estimate_node_cost(node: &ComputeNode, devices: usize) -> f64 {
    // FIXED: Clone the shape or use a fallback to satisfy the borrow checker
    let shape = node.output_shape.clone().unwrap_or_else(|| vec![1_i64]);
    let elements: i64 = shape.iter().product();
    
    let compute_cost = (elements as f64) / 1000.0;
    let parallel_cost = (compute_cost / devices as f64) + (devices as f64 * 0.5);
    
    parallel_cost
}

// Calculates the cost of a contiguous stage (layers i to j) on d devices.
fn stage_cost(layers: &[ComputeNode], start: usize, end: usize, devices: usize) -> f64 {
    let mut total_cost = 0.0;
    for i in start..=end {
        total_cost += estimate_node_cost(&layers[i], devices);
    }
    total_cost
}

// --- 4. The DP Solver ---

/// Inner DP: Minimizes the sum of stage latencies given a strict bottleneck limit.
/// Returns: Option<(Min Sum of Latencies, The Cuts Table)>
fn dp_inner(
    layers: &[ComputeNode],
    num_devices: usize,
    t_max_limit: f64,
) -> Option<(f64, Vec<Vec<(usize, usize)>>)> {
    let num_layers = layers.len();
    
    // dp[i][d] = min sum of latencies for first `i` layers using `d` devices
    let mut dp = vec![vec![f64::INFINITY; num_devices + 1]; num_layers + 1];
    
    // cuts[i][d] = (previous_layer_cut, previous_device_cut)
    let mut cuts = vec![vec![(0, 0); num_devices + 1]; num_layers + 1];
    
    dp[0][0] = 0.0;

    for i in 1..=num_layers {
        for d in 1..=num_devices {
            for j in 0..i {
                for k in 0..d {
                    let devices_for_stage = d - k;
                    let cost = stage_cost(layers, j, i - 1, devices_for_stage);

                    // Constraint: This stage cannot exceed our guessed bottleneck
                    if cost <= t_max_limit {
                        let total_cost = dp[j][k] + cost;
                        if total_cost < dp[i][d] {
                            dp[i][d] = total_cost;
                            cuts[i][d] = (j, k); // Record the decision for backtracing
                        }
                    }
                }
            }
        }
    }

    if dp[num_layers][num_devices] == f64::INFINITY {
        None 
    } else {
        Some((dp[num_layers][num_devices], cuts))
    }
}

/// Outer Loop: Binary searches the optimal bottleneck `t_max` to minimize overall pipeline latency.
pub fn find_optimal_pipeline(
    layers: &[ComputeNode],
    num_devices: usize,
    num_microbatches: f64,
) -> (f64, Vec<(usize, usize, usize)>) {
    let mut low = 0.0;
    let mut high = stage_cost(layers, 0, layers.len() - 1, 1); 
    
    let mut best_pipeline_latency = f64::INFINITY;
    let mut best_cuts: Option<Vec<Vec<(usize, usize)>>> = None;

    // Binary search over the bottleneck threshold
    for _ in 0..50 { 
        let current_t_max = (low + high) / 2.0;

        if let Some((min_sum_latencies, cuts)) = dp_inner(layers, num_devices, current_t_max) {
            let pipeline_latency = (num_microbatches - 1.0) * current_t_max + min_sum_latencies;
            
            if pipeline_latency < best_pipeline_latency {
                best_pipeline_latency = pipeline_latency;
                best_cuts = Some(cuts); // Save the winning path
            }
            
            high = current_t_max; // Try to squeeze the bottleneck tighter
        } else {
            low = current_t_max; // Bottleneck too strict, relax it
        }
    }

    // --- BACKTRACE THE DP TABLE ---
    let mut schedule = Vec::new();
    
    if let Some(cuts) = best_cuts {
        let mut curr_layer = layers.len();
        let mut curr_dev = num_devices;
        
        // Walk backward from the final state to the beginning
        while curr_layer > 0 && curr_dev > 0 {
            let (prev_layer, prev_dev) = cuts[curr_layer][curr_dev];
            let dev_for_stage = curr_dev - prev_dev;
            
            // Record this stage: (start_index, end_index, devices_assigned)
            schedule.push((prev_layer, curr_layer - 1, dev_for_stage));
            
            curr_layer = prev_layer;
            curr_dev = prev_dev;
        }
        
        schedule.reverse(); // Flip to chronological order
    }

    (best_pipeline_latency, schedule)
}

fn main() {
    // 1. Initialize Logger
    let env = env_logger::Env::new().filter("PHANTORA_LOG");
    env_logger::init_from_env(env);

    // 2. Get CLI Arguments (including --graph)
    let args = phantora::args::get_args();
    let graph_path = &args.graph;

    // 3. Read the REAL JSON file from Python
    let graph_json = std::fs::read_to_string(graph_path)
        .expect("Failed to read the compute_graph.json file");
    
    let real_graph: ComputeGraph = serde_json::from_str(&graph_json)
        .expect("Failed to parse JSON into ComputeNode structs");

    println!("Loaded {} nodes from {}", real_graph.len(), graph_path.display());

    // 4. Topological Sort (Crucial for ResNet/Transformer blocks)
    let sorted_layers = topological_sort(&real_graph);
    println!("Compute layers after topological sort: {}", sorted_layers.len());

    // 5. Run the Alpa DP Search
    let num_gpus = 4;        // Optimizing for 4 GPUs
    let microbatches = 64.0; // High count to favor pipelining
    
    let (optimal_time, schedule) = find_optimal_pipeline(&sorted_layers, num_gpus, microbatches);
    
    println!("\n========================================");
    println!("🏆 OPTIMAL ALPA EXECUTION PLAN");
    println!("========================================");
    println!("Total Pipeline Latency: {:.2} ms", optimal_time);
    println!("----------------------------------------");
    
    for (stage_idx, (start, end, devs)) in schedule.iter().enumerate() {
        let layer_names: Vec<String> = sorted_layers[*start..=*end]
            .iter()
            .map(|n| n.name.clone())
            .collect();
            
        println!(
            "Stage {} | GPUs Allocated: {} | Layers: {:?}", 
            stage_idx, devs, layer_names
        );
    }
    println!("========================================\n");
}