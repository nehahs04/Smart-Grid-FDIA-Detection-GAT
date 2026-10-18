"""
Attention Weight Visualization & Explainability Analysis for Smart Grid FDIA GAT
Visualizes spatial attention shift across IEEE 14-bus transmission lines under attack.
Author: Neha H S (MAHE-ISAC Cybersecurity Research)
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from grid_topology import IEEE14BusGrid
from model import SmartGridGAT

def analyze_attention_shifts(output_plot: str = "attention_analysis.png"):
    print("=" * 65)
    print("  Smart Grid GAT - Attention Mechanism Interpretability (XAI)")
    print("  Author: Neha H S | MAHE-ISAC Cybersecurity Research")
    print("=" * 65)

    grid = IEEE14BusGrid()
    edge_index = []
    for u, v in grid.edges:
        edge_index.append([u, v])
        edge_index.append([v, u])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Instantiate model
    model = SmartGridGAT(in_channels=4, hidden_dim=32, num_heads=4, dropout=0.0)
    model.eval()

    # Generate baseline healthy state and simulated attack
    healthy = grid.generate_healthy_state()
    attacked = grid.inject_fdia_attack(healthy, attack_intensity=0.35)

    x_healthy = torch.tensor(healthy, dtype=torch.float)
    x_attack = torch.tensor(attacked, dtype=torch.float)

    with torch.no_grad():
        _, attn_healthy = model(x_healthy, edge_index)
        _, attn_attack = model(x_attack, edge_index)

    # Reconstruct 14x14 attention adjacency matrices
    num_nodes = grid.num_nodes
    mat_healthy = np.zeros((num_nodes, num_nodes))
    mat_attack = np.zeros((num_nodes, num_nodes))

    src_nodes = edge_index[0].numpy()
    dst_nodes = edge_index[1].numpy()
    w_h = attn_healthy.numpy()
    w_a = attn_attack.numpy()

    for i in range(len(src_nodes)):
        u, v = src_nodes[i], dst_nodes[i]
        mat_healthy[u, v] = w_h[i]
        mat_attack[u, v] = w_a[i]

    delta = np.abs(mat_attack - mat_healthy)

    # Plotting Interpretability Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(mat_healthy, cmap="Blues", interpolation="nearest")
    axes[0].set_title("Normal State: Bus Attention", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Target Bus Index")
    axes[0].set_ylabel("Source Bus Index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mat_attack, cmap="Reds", interpolation="nearest")
    axes[1].set_title("FDIA Attack: Anomaly Attention", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Target Bus Index")
    axes[1].set_ylabel("Source Bus Index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(delta, cmap="magma", interpolation="nearest")
    axes[2].set_title("Attention Shift (|Attack - Normal|)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Target Bus Index")
    axes[2].set_ylabel("Source Bus Index")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Explainable AI (XAI): IEEE 14-Bus Topological Attention Coefficients | Neha H S", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"\n[OK] Explainability visualization successfully exported to: '{output_plot}'")
    print(f"[+] Max Topological Attention Perturbation: {delta.max():.4f}")

if __name__ == "__main__":
    analyze_attention_shifts()
