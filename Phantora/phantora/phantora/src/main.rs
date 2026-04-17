use phantora::alpa::{ComputeGraph, topological_sort, find_optimal_pipeline};

fn main() {
    // 1. Initialize Logger
    let env = env_logger::Env::new().filter("PHANTORA_LOG");
    env_logger::init_from_env(env);

    // 2. Get CLI Arguments
    let args = phantora::args::get_args();
    let graph_path = &args.graph;

    // 3. Read the JSON file
    let graph_json = std::fs::read_to_string(graph_path)
        .expect("Failed to read the compute_graph.json file");
    
    let real_graph: ComputeGraph = serde_json::from_str(&graph_json)
        .expect("Failed to parse JSON into ComputeNode structs");

    println!("Loaded {} nodes from {}", real_graph.len(), graph_path.display());

    // 4. Topological Sort
    let sorted_layers = topological_sort(&real_graph);
    println!("Compute layers after topological sort: {}", sorted_layers.len());

    // 5. Run the Alpa DP + ILP Search
    let num_gpus = 4;
    let microbatches = 64.0; 
    
    let (best_pipeline_latency, schedule) = find_optimal_pipeline(&sorted_layers, num_gpus, microbatches);
    
    println!("\n========================================");
    println!("🏆 OPTIMAL ALPA EXECUTION PLAN");
    println!("========================================");
    println!("Total Pipeline Latency: {:.2} ms", best_pipeline_latency);
    println!("----------------------------------------");
    
    for (stage_idx, (start, end, devs)) in schedule.iter().enumerate() {
        let layer_names: Vec<String> = sorted_layers[*start..=*end]
            .iter()
            .map(|n| n.name.clone())
            .collect();
            
        println!("Stage {} | GPUs Allocated: {}", stage_idx, devs);
        println!("  Layers ({} total):", layer_names.len());
        
        // Pretty print the layers by breaking them into chunks of 5 per line
        for chunk in layer_names.chunks(5) {
            println!("    {}", chunk.join(", "));
        }
        println!("----------------------------------------");
    }
    println!("========================================\n");
}