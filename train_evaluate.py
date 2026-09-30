"""
Training, Validation, and Metric Evaluation Script for Smart Grid FDIA GAT
Author: Neha H S (MAHE-ISAC Cybersecurity Research)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from grid_topology import IEEE14BusGrid
from model import SmartGridGAT

def train_and_evaluate(epochs: int = 40, lr: float = 0.003):
    print("=" * 65)
    print("Smart Grid False Data Injection Attack (FDIA) Detection via GAT")
    print("Author: Neha H S | Research at MAHE-ISAC CoE for Cybersecurity")
    print("=" * 65)

    # 1. Generate Dataset
    print("\n[+] Generating IEEE 14-Bus cyber-physical grid simulation data...")
    grid = IEEE14BusGrid()
    X_list, y_list, edge_index = grid.generate_dataset(num_samples=1200, attack_ratio=0.5)

    # Train / Test split (80 / 20)
    split_idx = int(0.8 * len(X_list))
    train_X, test_X = X_list[:split_idx], X_list[split_idx:]
    train_y, test_y = y_list[:split_idx], y_list[split_idx:]

    print(f"Training Samples: {len(train_X)} | Test Samples: {len(test_X)}")

    # 2. Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartGridGAT(in_channels=4, hidden_dim=32, num_heads=4, dropout=0.15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    edge_index = edge_index.to(device)
    
    # 3. Training Loop
    print("\n[+] Training Graph Attention Network...")
    loss_history = []

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # Shuffle indices
        indices = np.random.permutation(len(train_X))

        for idx in indices:
            x = train_X[idx].to(device)
            target = train_y[idx:idx+1].to(device)

            optimizer.zero_grad()
            logits, _ = model(x, edge_index)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_X)
        loss_history.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f}")

    # 4. Evaluation
    print("\n[+] Evaluating Model on Test Grid Topologies...")
    model.eval()
    y_preds = []
    y_probs = []

    with torch.no_grad():
        for i in range(len(test_X)):
            x = test_X[i].to(device)
            logits, attn = model(x, edge_index)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = np.argmax(probs)
            
            y_preds.append(pred)
            y_probs.append(probs[1])

    y_true = test_y.numpy()
    y_preds = np.array(y_preds)
    y_probs = np.array(y_probs)

    acc = accuracy_score(y_true, y_preds)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_preds)

    print("\n" + "=" * 45)
    print("           EXPERIMENTAL RESULTS")
    print("=" * 45)
    print(f"Accuracy:   {acc * 100:.2f}%")
    print(f"ROC-AUC:    {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [TN: {cm[0,0]}  FP: {cm[0,1]}]")
    print(f"  [FN: {cm[1,0]}  TP: {cm[1,1]}]")
    print("\nClassification Report:")
    print(classification_report(y_true, y_preds, target_names=["Healthy State", "FDIA Anomaly"]))

    # Save metrics plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='#3b82f6', label='GAT Cross-Entropy Loss')
    plt.title("FDIA Detection Training Loss over Epochs", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("[+] Saved training progression curve to 'loss_curve.png'")

    return acc, auc

if __name__ == "__main__":
    train_and_evaluate(epochs=20, lr=0.003)
