import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

# ======================================================================================
# Define input/output dimensions for the GNN
# inputs  = number of input node features (e.g., 60)
# hidden  = size of hidden layers in MLPs
# outputs = number of output classes (e.g., 2: QCD vs Higgs)
# ======================================================================================
inputs = 74
hidden = 256
outputs = 2

# ======================================================================================
# EdgeBlock: Computes edge features using source and destination node features
# ======================================================================================
class EdgeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()  # standard parent class init

        self.edge_mlp = Seq(
            Lin(inputs * 2, hidden),         # Input: concatenated src + dest
            BatchNorm1d(hidden),
            ReLU(),
            Lin(hidden, hidden),             # Further transform
            BatchNorm1d(hidden)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest], dim=1)  # shape: [n_edges, 2*inputs]
        return self.edge_mlp(out)            # shape: [n_edges, hidden]

# ======================================================================================
# NodeBlock: Updates node features using incoming edge messages
# ======================================================================================
class NodeBlock(torch.nn.Module):
    def __init__(self):
        super(NodeBlock, self).__init__()

        self.node_mlp_1 = Seq(
            Lin(inputs + hidden, hidden),    # Combine current node and incoming edge msg
            BatchNorm1d(hidden),
            ReLU(),
            Lin(hidden, hidden),
            BatchNorm1d(hidden)
        )

        self.node_mlp_2 = Seq(
            Lin(inputs + hidden, hidden),    # Again combine and transform
            BatchNorm1d(hidden),
            ReLU(),
            Lin(hidden, hidden),
            BatchNorm1d(hidden)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index                # row: source, col: destination
        out = torch.cat([x[row], edge_attr], dim=1)  # gather source node and edge_attr
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))  # aggregate messages at each node
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

# ======================================================================================
# GlobalBlock: Pool node-level features to form a graph-level representation
# ======================================================================================
class GlobalBlock(torch.nn.Module):
    def __init__(self):
        super(GlobalBlock, self).__init__()

        self.global_mlp = Seq(
            Lin(hidden, hidden),
            BatchNorm1d(hidden),
            ReLU(),
            Lin(hidden, outputs)             # Output layer: logits for 2 classes
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter_mean(x, batch, dim=0)  # global mean pooling over graph
        return self.global_mlp(out)          # returns graph-level predictions (logits)

# ======================================================================================
# Full MetaLayer-based GNN model using PyTorch Geometric's MetaLayer
# ======================================================================================
class InteractionNetwork(torch.nn.Module):
    def __init__(self):
        super(InteractionNetwork, self).__init__()

        # Wrap edge/node/global blocks into one MetaLayer
        self.interactionnetwork = MetaLayer(
            EdgeBlock(),
            NodeBlock(),
            GlobalBlock()
        )

        self.bn = BatchNorm1d(inputs)  # Initial batch norm on input features

    def forward(self, x, edge_index, batch):
        x = self.bn(x)  # normalize node input features
        x, edge_attr, u = self.interactionnetwork(x, edge_index, None, None, batch)
        return u  # `u` is the graph-level output from GlobalBlock
