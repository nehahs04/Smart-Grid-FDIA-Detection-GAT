<h1 align="center">Smart Grid FDIA Detection using Graph Attention Networks (GAT) ⚡🛡️</h1>
<h3 align="center">Cyber-Physical Anomaly Detection for False Data Injection Attacks in Power Transmission Grids</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Research-MAHE--ISAC%20CoE-blue?style=for-the-badge" alt="MAHE-ISAC CoE" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Graph%20Neural%20Networks-GAT-6366F1?style=for-the-badge" alt="GNN" />
  <img src="https://img.shields.io/badge/Cybersecurity-Smart%20Grid-10B981?style=for-the-badge" alt="Cybersecurity" />
</p>

---

## 📌 Research Background & Motivation

In modern cyber-physical smart grids, Supervisory Control and Data Acquisition (SCADA) systems rely on State Estimation (SE) to monitor bus voltages, phase angles, and transmission line power flows. However, adversaries can orchestrate **False Data Injection Attacks (FDIA)**:
$$\mathbf{z}_{bad} = \mathbf{z} + \mathbf{a}$$
Where $\mathbf{a} = \mathbf{H}\mathbf{c}$. Under this attack formulation, the attack vector $\mathbf{a}$ lies in the column space of the grid measurement matrix $\mathbf{H}$. Consequently, traditional Euclidean residual / $\chi^2$ Bad Data Detection (BDD) tests fail:
$$||\mathbf{z}_{bad} - \mathbf{H}\hat{\mathbf{x}}_{bad}|| = ||\mathbf{z} - \mathbf{H}\hat{\mathbf{x}}|| \le \tau$$

This research, conducted by **Neha H S** at the **MAHE–ISAC Centre of Excellence for Cybersecurity, Manipal**, utilizes **Graph Attention Networks (GATs)** to learn spatial correlations and dynamic edge attention weights across power grid bus topologies, pinpointing stealthy coordinate perturbations that evade classical statistical filters.

---

## 🏗️ Architecture Overview

```
IEEE 14-Bus Grid Topology & SCADA Telemetry
[Voltage Magnitude | Phase Angle | Active Power P | Reactive Power Q]
                           │
                           ▼
          Multi-Head Graph Attention Layer 1
           (K=4 heads, Attention Coefficients α_ij)
                           │
                           ▼
                  LeakyReLU + ELU Activation
                           │
                           ▼
             Graph Attention Layer 2 (Hidden 32)
                           │
                           ▼
             Global Mean Graph Pooling Readout
                           │
                           ▼
          MLP Classifier with Dropout Regularization
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Healthy State (0)           FDIA Anomaly (1)
```

---

## 🔬 GAT Mathematical Formulation

For each electrical bus $i$ and adjacent bus $j \in \mathcal{N}_i$, the attention coefficient $\alpha_{ij}$ is computed as:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \,||\, \mathbf{W}h_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \,||\, \mathbf{W}h_k]\right)\right)}$$

The aggregated node representation with multi-head attention ($K$ heads) is:
$$h_i' = \Vert_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k h_j\right)$$

---

## 📊 Experimental Results

Tested on simulated IEEE 14-bus cyber-physical grid states:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **96.25%** |
| **ROC-AUC** | **0.984** |
| **Precision (FDIA)** | **0.95** |
| **Recall (FDIA)** | **0.97** |
| **F1-Score** | **0.96** |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/nehahs04/Smart-Grid-FDIA-Detection-GAT.git
cd Smart-Grid-FDIA-Detection-GAT
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training & Evaluation
```bash
python train_evaluate.py
```
This script will:
- Simulate IEEE 14-bus healthy states and stealthy FDIA perturbations.
- Train the multi-head GAT model.
- Print detailed classification reports and confusion matrices.
- Export the training loss convergence graph to `loss_curve.png`.

---

## 👩‍💻 Author & Research Attribution

**Neha H S**  
- **Role:** Cybersecurity Research Intern (Jan 2026 – Feb 2026)  
- **Host Institution:** MAHE–ISAC Centre of Excellence for Cybersecurity, Manipal  
- **Academic Program:** B.E. Information Science and Engineering, Global Academy of Technology  
- **LinkedIn:** [neha-h-s-4701b0375](https://www.linkedin.com/in/neha-h-s-4701b0375/)  
- **GitHub:** [@nehahs04](https://github.com/nehahs04)  
