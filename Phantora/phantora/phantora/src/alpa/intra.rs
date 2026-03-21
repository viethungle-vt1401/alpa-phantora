use super::graph::*;

#[derive(Clone)]
pub struct Strategy {
    pub partition: usize, // number of shards
    pub cost: f64,
}

pub fn solve_intra(graph: &ModelGraph, devices: usize) -> Vec<Strategy> {
    graph.nodes.iter().map(|node| {
        // VERY IMPORTANT:
        // This is where Alpa normally tries many sharding strategies

        let mut best = Strategy {
            partition: 1,
            cost: estimate(node, 1),
        };

        for p in [1, 2, 4, 8] {
            if p > devices { continue; }

            let cost = estimate(node, p);
            if cost < best.cost {
                best = Strategy { partition: p, cost };
            }
        }

        best
    }).collect()
}

fn estimate(node: &OpNode, partition: usize) -> f64 {
    // simple proxy:
    // compute / partition + communication penalty

    let base = 1.0;
    let compute = base / partition as f64;
    let comm = (partition as f64).ln();

    compute + comm
}