# HedonicMFG-FL: A Bi-Level Game-Theoretic Framework for Heterogeneous Federated Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)


> **HedonicMFG-FL** addresses data heterogeneity in federated learning through a novel bi-level game-theoretic framework. At the upper level, a **hedonic game** partitions clients into Nash-stable coalitions. At the lower level, a **mean field game** optimises each client's training strategy within its coalition. A private **personalisation head** per client further adapts predictions to local distributions.

---

## 📁 Repository Structure

```text
HedonicMFG-FL/
│
├── ablation_runs/              # Ablation study experiments
│   ├── alpha_ablation/         # Effect of Dirichlet α on MNIST & FashionMNIST
│   │   ├── runner_mnist.py     # Runner for MNIST α ∈ {0.05, 0.1, 0.3, 0.5, 1.0}
│   │   └── runner_fmnist.py    # Runner for FashionMNIST α ablation
│   ├── k_ablation/             # Effect of number of coalitions K on MNIST
│   │   └── runner_k.py         # Runner for K ∈ {2, 3, 4, 5, 6, 8}
│   └── client_ablation/        # Effect of federation size N on FashionMNIST
│       └── runner_clients.py   # Runner for N ∈ {10, 20, 30, 50, 75, 100}
│
├── adult_census_runs/          # Adult Census dataset experiments
│   ├── runner_adult.py         # Main experiment runner (25 clients, 75 rounds)
│   ├── algorithms/             # Algorithm implementations
│   ├── data.py                 # Data loading & Dirichlet partitioning
│   ├── models.py               # MLP model for tabular classification
│   └── results/                # Saved results, plots, JSON logs
│
├── fashion_mnist_runs/         # FashionMNIST dataset experiments
│   ├── runner_fmnist.py        # Main experiment runner (20 clients, 50 rounds)
│   ├── algorithms/             # Algorithm implementations
│   ├── data.py                 # Data loading & Dirichlet partitioning
│   ├── models.py               # CNN model for image classification
│   └── results/                # Saved results, plots, JSON logs
│
├── imdb_runs/                  # IMDB Reviews dataset experiments
│   ├── runner_imdb.py          # Main experiment runner (20 clients, 50 rounds)
│   ├── algorithms/             # Algorithm implementations
│   ├── data.py                 # Data loading, tokenisation, Dirichlet partitioning
│   ├── models.py               # TextCNN model for sentiment classification
│   └── results/                # Saved results, plots, JSON logs
│
├── mnist_runs/                 # MNIST dataset experiments
│   ├── runner.py               # Main experiment runner (20 clients, 50 rounds)
│   ├── algorithms/             # Algorithm implementations
│   ├── data.py                 # Data loading & Dirichlet partitioning
│   ├── models.py               # CNN model for digit classification
│   └── results/                # Saved results, plots, JSON logs
│
├── extras/                     # Utility scripts and visualisation tools
│   ├── cifar-10                # CIFAR-10 Codes  
│
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

1. Clone the Repository
cd HedonicMFG-FL

2. Install Dependencies
pip install -r requirements.txt

Requirements: Python 3.8+, PyTorch 1.12+, CUDA (recommended). See requirements.txt for the full list.

3. Run an Experiment

Each dataset folder has its own self-contained runner. From the repository root:

#### MNIST (20 clients, 50 rounds, K=3 coalitions)
cd mnist_runs
python runner.py

#### FashionMNIST
cd fashion_mnist_runs
python runner_fmnist.py

#### Adult Census (25 clients, 75 rounds, K=4 coalitions)
cd adult_census_runs
python runner_adult.py

#### IMDB Reviews (20 clients, 50 rounds, K=3 coalitions)
cd imdb_runs
python runner_imdb.py

Results (JSON logs + PNG plots) are saved automatically to each folder's results/ directory.



## 🧪 Ablation Studies

All ablation experiments are in ablation_runs/. Each runner accepts a command-line argument for the parameter being varied.

Effect of Dirichlet α (heterogeneity severity)
cd ablation_runs/alpha_ablation

#### MNIST — vary α
python runner_mnist.py --alpha 0.05
python runner_mnist.py --alpha 0.1
python runner_mnist.py --alpha 0.3
python runner_mnist.py --alpha 0.5
python runner_mnist.py --alpha 1.0

#### FashionMNIST — vary α
python runner_fmnist.py --alpha 0.05
python runner_fmnist.py --alpha 0.1
python runner_fmnist.py --alpha 0.3
python runner_fmnist.py --alpha 0.5
python runner_fmnist.py --alpha 1.0
Effect of Number of Coalitions K
cd ablation_runs/k_ablation

python runner_k.py --k 2
python runner_k.py --k 3
python runner_k.py --k 4
python runner_k.py --k 5
python runner_k.py --k 6
python runner_k.py --k 8
Effect of Number of Clients N
cd ablation_runs/client_ablation

python runner_clients.py --clients 10
python runner_clients.py --clients 20
python runner_clients.py --clients 30
python runner_clients.py --clients 50
python runner_clients.py --clients 75
python runner_clients.py --clients 100




## ⚙️ Default Hyperparameters
Parameter	Value
Local learning rate	0.01
Batch size	128
Local epochs per round	3
Optimiser	SGD
Dirichlet α (main runs)	0.3
Random seed	42
Sinkhorn regularisation ε	0.1
MFG sync penalty γ_sync	0.1 (0.3 for Adult Census)
MFG fairness weight λ_fair	0.5–0.8 (dataset-dependent)
MFG convergence tolerance	1e-3
Max MFG iterations	5




## 🔬 How HedonicMFG-FL Works
Phase 1 — FedAvg Warmup (T_w rounds)
   

Phase 2 — Hedonic Coalition Formation
    

Phase 3 — MFG-Based Clustered Training (T - T_w rounds)


    
## 📬 Contact

Varun Kukreti — 2024csm1020@iitrpr.ac.in

Supervisor: Dr. Shweta Jain — shwetajain@iitrpr.ac.in

Department of Computer Science & Engineering, IIT Ropar
