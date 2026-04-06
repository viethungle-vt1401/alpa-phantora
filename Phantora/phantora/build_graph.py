import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import json
import argparse


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_proj = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp_up = nn.Linear(dim, dim * 4)
        self.relu = nn.ReLU()
        self.mlp_down = nn.Linear(dim * 4, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn_proj(self.ln1(x))
        x = x + self.mlp_down(self.relu(self.mlp_up(self.ln2(x))))
        return x


class DeepModel(nn.Module):
    def __init__(self, hidden_size, num_layers=24):
        super().__init__()
        self.embed = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, 1000)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


def extract_graph(hidden_size=2048, batch_size=32768):
    model = DeepModel(hidden_size, num_layers=24)
    model.eval()

    traced = symbolic_trace(model)

    # Use the dynamic batch size and hidden size for the dummy input
    dummy = torch.randn(batch_size, hidden_size)

    # Prevent PyTorch from saving intermediate tensors for backprop
    with torch.no_grad():
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
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Extract PyTorch compute graph for Phantora.")
    parser.add_argument("--output", type=str, default="compute_graph.json", help="Output JSON file path")
    parser.add_argument("--batch-size", type=int, default=32768, help="Batch size for the dummy input")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden size for the model")
    args = parser.parse_args()

    # Pass the arguments to the extraction function
    nodes = extract_graph(hidden_size=args.hidden_size, batch_size=args.batch_size)

    print("\n" + "="*80)
    print(f"{'IDX':<5} | {'OPERATION':<18} | {'OUTPUT SHAPE':<25} | {'NODE NAME'}")
    print("="*80)

    for i, node in enumerate(nodes):
        shape_str = str(node['output_shape']) if node['output_shape'] else "None"
        print(f"{i:<5} | {node['op']:<18} | {shape_str:<25} | {node['name']}")

    print("="*80)

    # Write to the dynamically provided output file
    with open(args.output, "w") as f:
        json.dump(nodes, f, indent=4)

    print(f"\nSuccessfully exported {len(nodes)} nodes to {args.output}")
