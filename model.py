"""
Graph Attention Network (GAT) Architecture for FDIA Anomaly Detection
Author: Neha H S (MAHE-ISAC Cybersecurity Research)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer (Velickovic et al.)
    Computes dynamic attention coefficients between connected electrical buses.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        # Linear projection: h -> Wh
        Wh = self.W(h) # (N, out_features)
        N = Wh.size(0)

        # Extract edges
        src, dst = edge_index[0], edge_index[1]

        # Calculate self-attention coefficients on edges
        Wh_src = Wh[src]
        Wh_dst = Wh[dst]
        edge_cat = torch.cat([Wh_src, Wh_dst], dim=1) # (E, 2*out_features)
        edge_scores = self.leaky_relu(torch.matmul(edge_cat, self.a).squeeze(1)) # (E)

        # Softmax over neighborhood
        # Use numerically stable edge-level softmax
        exp_scores = torch.exp(edge_scores - edge_scores.max())
        
        # Aggregate denominator per node
        denom = torch.zeros(N, device=h.device)
        denom.scatter_add_(0, dst, exp_scores)
        
        attn_weights = exp_scores / (denom[dst] + 1e-8)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Message passing & aggregation
        weighted_features = Wh_src * attn_weights.unsqueeze(1)
        h_prime = torch.zeros(N, self.out_features, device=h.device)
        h_prime.scatter_add_(0, dst.unsqueeze(1).expand_as(weighted_features), weighted_features)

        return F.elu(h_prime), attn_weights

class SmartGridGAT(nn.Module):
    """End-to-End Smart Grid FDIA Anomaly Detection Network
    Stack of Multi-Head GAT layers followed by global graph pooling and MLP classifier.
    """
    def __init__(self, in_channels: int = 4, hidden_dim: int = 32, num_heads: int = 4, dropout: float = 0.2):
        super(SmartGridGAT, self).__init__()
        
        # Multi-Head GAT Layer 1
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_channels, hidden_dim // num_heads, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        # GAT Layer 2
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
        
        # Graph-level Readout & Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2) # Binary classification: [Healthy, FDIA Anomaly]
        )

    def forward(self, x, edge_index):
        # Layer 1: Concatenate multi-head attention outputs
        head_outputs = [head(x, edge_index)[0] for head in self.heads]
        h1 = torch.cat(head_outputs, dim=1)
        
        # Layer 2
        h2, attn_weights = self.gat2(h1, edge_index)
        
        # Global mean pooling across all buses to get graph representation
        graph_embedding = torch.mean(h2, dim=0, keepdim=True)
        
        # Output logits
        logits = self.classifier(graph_embedding)
        return logits, attn_weights

if __name__ == "__main__":
    from grid_topology import IEEE14BusGrid
    grid = IEEE14BusGrid()
    X_list, y_list, edge_index = grid.generate_dataset(num_samples=2)
    
    model = SmartGridGAT(in_channels=4, hidden_dim=32, num_heads=4)
    out, attn = model(X_list[0], edge_index)
    print("Model forward pass successful!")
    print(f"Logits shape: {out.shape}")
    print(f"Attention weights shape: {attn.shape}")
