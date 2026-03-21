use crate::torch_call::TorchCall;

#[derive(Clone)]
pub struct OpNode {
    pub id: usize,
    pub call: TorchCall,
}

#[derive(Clone)]
pub struct ModelGraph {
    pub nodes: Vec<OpNode>,
}

pub fn build_graph(calls: &[TorchCall]) -> ModelGraph {
    let nodes = calls
        .iter()
        .enumerate()
        .map(|(i, c)| OpNode {
            id: i,
            call: c.clone(),
        })
        .collect();

    ModelGraph { nodes }
}