use super::graph::*;
use super::intra::Strategy;
use super::Stage;

pub fn solve_inter(
    graph: &ModelGraph,
    strategies: &[Strategy],
    devices: usize,
) -> Vec<Stage> {

    let n = graph.nodes.len();
    let mut dp = vec![f64::INFINITY; n + 1];
    let mut split = vec![0; n + 1];

    dp[0] = 0.0;

    for i in 1..=n {
        for j in 0..i {
            let cost = dp[j] + stage_cost(&strategies[j..i]);
            if cost < dp[i] {
                dp[i] = cost;
                split[i] = j;
            }
        }
    }

    // reconstruct
    let mut stages = vec![];
    let mut i = n;

    while i > 0 {
        let j = split[i];

        stages.push(Stage {
            ops: (j..i).collect(),
            devices: (0..devices).collect(),
        });

        i = j;
    }

    stages.reverse();
    stages
}

fn stage_cost(strats: &[Strategy]) -> f64 {
    let compute: f64 = strats.iter().map(|s| s.cost).sum();
    // 🔥 key idea: penalize large stages
    let size_penalty = (strats.len() as f64).powi(2) * 0.5;

    compute + size_penalty
}