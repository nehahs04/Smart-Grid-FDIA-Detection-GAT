"""
Smart Grid Power Topology & FDIA Perturbation Generator
Simulates IEEE 14-bus Power Network with stealthy False Data Injection Attacks (FDIA).
Author: Neha H S (MAHE-ISAC Cybersecurity Research)
"""

import numpy as np
import networkx as nx
import torch

class IEEE14BusGrid:
    def __init__(self):
        # IEEE 14-bus standard transmission lines (0-indexed bus pairs)
        self.edges = [
            (0, 1), (0, 4), (1, 2), (1, 3), (1, 4),
            (2, 3), (3, 4), (3, 6), (3, 8), (4, 5),
            (5, 10), (5, 11), (5, 12), (6, 7), (6, 8),
            (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)
        ]
        self.num_nodes = 14
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))
        self.graph.add_edges_from(self.edges)

    def generate_healthy_state(self):
        """Generates realistic normal baseline measurements for 14 buses:
        Features: [Voltage Magnitude (pu), Phase Angle (rad), Active Power P (MW), Reactive Power Q (MVAR)]
        """
        v_mag = np.random.normal(1.02, 0.02, self.num_nodes) # around 1.0 pu
        v_ang = np.random.normal(0.0, 0.08, self.num_nodes)  # around 0 rad
        p_inj = np.random.normal(0.45, 0.15, self.num_nodes) # active power
        q_inj = np.random.normal(0.18, 0.06, self.num_nodes) # reactive power
        
        node_features = np.column_stack([v_mag, v_ang, p_inj, q_inj])
        return node_features

    def inject_fdia_attack(self, healthy_features, attack_intensity: float = 0.25):
        """Injects a stealthy False Data Injection Attack (FDIA).
        Perturbs voltage angles and active power injections in a coordinated topological sub-network,
        mimicking a linearized state estimation attack: a = H * c
        """
        attacked_features = healthy_features.copy()
        
        # Attacker targets a coordinated subset of connected buses (e.g. buses 1, 2, 3, 4)
        target_buses = np.random.choice(self.num_nodes, size=np.random.randint(3, 6), replace=False)
        
        # Coordinated stealth attack vector c
        angle_shift = np.random.uniform(0.15, attack_intensity, size=len(target_buses))
        power_shift = angle_shift * np.random.uniform(1.2, 2.0, size=len(target_buses))
        
        attacked_features[target_buses, 1] += angle_shift
        attacked_features[target_buses, 2] += power_shift
        
        return attacked_features

    def generate_dataset(self, num_samples: int = 1000, attack_ratio: float = 0.5):
        """Generates a graph-based dataset of healthy and attacked grid instances."""
        X_list = []
        y_list = []
        
        # Edge index for PyTorch (2, E)
        edge_index = []
        for u, v in self.edges:
            edge_index.append([u, v])
            edge_index.append([v, u]) # Bidirectional transmission line
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        for _ in range(num_samples):
            healthy = self.generate_healthy_state()
            is_attack = np.random.rand() < attack_ratio
            
            if is_attack:
                sample_x = self.inject_fdia_attack(healthy)
                label = 1 # FDIA Anomaly
            else:
                sample_x = healthy
                label = 0 # Healthy
                
            X_list.append(torch.tensor(sample_x, dtype=torch.float))
            y_list.append(label)

        return X_list, torch.tensor(y_list, dtype=torch.long), edge_index

if __name__ == "__main__":
    grid = IEEE14BusGrid()
    X, y, edge_idx = grid.generate_dataset(num_samples=10)
    print(f"IEEE 14-Bus Grid Initialized.")
    print(f"Sample node feature shape: {X[0].shape} (Buses: 14, Features: 4)")
    print(f"Edge index shape: {edge_idx.shape} (Directed edges: {edge_idx.shape[1]})")
    print(f"Sample labels (0=Healthy, 1=FDIA): {y.tolist()}")
