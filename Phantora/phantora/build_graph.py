import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import json

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Simplified Attention-like Projection
        self.attn_proj = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        
        # MLP Block (The heavy compute part)
        self.mlp_up = nn.Linear(dim, dim * 4)
        self.relu = nn.ReLU()
        self.mlp_down = nn.Linear(dim * 4, dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Sublayer 1: Attention skip
        x = x + self.attn_proj(self.ln1(x))
        # Sublayer 2: MLP skip
        x = x + self.mlp_down(self.relu(self.mlp_up(self.ln2(x))))
        return x

class DeepModel(nn.Module):
    def __init__(self, hidden_size, num_layers=24):
        super().__init__()
        self.embed = nn.Linear(hidden_size, hidden_size)
        # Stack 24 layers to create massive depth
        self.layers = nn.ModuleList([TransformerBlock(hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, 1000)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x

def extract_graph():
    hidden_size = 2048 # Large enough to make communication costs matter
    model = DeepModel(hidden_size, num_layers=24)
    model.eval()
    
    # Trace the 24-layer stack
    traced = symbolic_trace(model)
    
    # Batch size 64 to increase the compute-to-comm ratio
    dummy = torch.randn(64, hidden_size)
    ShapeProp(traced).propagate(dummy)

    graph_nodes = []
    for node in traced.graph.nodes:
        out_shape = list(node.meta['tensor_meta'].shape) if 'tensor_meta' in node.meta else None
        graph_nodes.append({
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": [str(arg) for arg in node.args],
            "output_shape": out_shape
        })
    return graph_nodes

if __name__ == "__main__":
    nodes = extract_graph()
    with open("compute_graph.json", "w") as f:
        json.dump(nodes, f, indent=4)
    print(f"Exported {len(nodes)} nodes for a 24-layer Deep Model.")