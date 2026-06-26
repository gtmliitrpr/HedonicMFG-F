"""
algorithms/ifca.py — IFCA baseline.
From: Ghosh et al. 2020, 'An Efficient Framework for Clustered Federated Learning'
Each client selects the cluster model with lowest local loss, trains on it,
and the server aggregates within each cluster.
"""

import torch
import copy
import time
import numpy as np
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from local_trainer import local_train_standard
from models import get_model


def run_ifca(config, client_train_loaders, client_val_loaders,
             global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: IFCA")
    print("="*50)

    tracker = ResultsTracker("IFCA")
    num_clients = config["num_clients"]
    total_rounds = config["total_rounds"]
    K = config["ifca_num_clusters"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]

    # Initialize K cluster models
    cluster_models = [get_model(config).to(device) for _ in range(K)]

    def estimate_loss(model, loader):
        """Estimate loss on first batch of loader."""
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                return criterion(out, y).item()
        return float("inf")

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()

        # Each client selects best cluster
        client_clusters = []
        for i in range(num_clients):
            losses = [estimate_loss(cluster_models[k], client_train_loaders[i])
                      for k in range(K)]
            best_k = int(np.argmin(losses))
            client_clusters.append(best_k)

        # Train each client on their selected cluster model
        cluster_updates = {k: [] for k in range(K)}
        cluster_weights = {k: [] for k in range(K)}

        for i in range(num_clients):
            k = client_clusters[i]
            local_model = copy.deepcopy(cluster_models[k])
            local_model = local_train_standard(
                local_model, client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device
            )
            cluster_updates[k].append(local_model)
            cluster_weights[k].append(client_data_sizes[i])

        # Aggregate within each cluster
        for k in range(K):
            if cluster_updates[k]:
                total_w = sum(cluster_weights[k])
                norm_w = [w / total_w for w in cluster_weights[k]]
                agg_params = fedavg_aggregate(cluster_updates[k], norm_w)
                for p, ap in zip(cluster_models[k].parameters(), agg_params):
                    p.data.copy_(ap)

        # Global model = weighted average of cluster models
        cluster_sizes = [sum(cluster_weights[k]) for k in range(K)]
        total_size = sum(cluster_sizes)
        if total_size > 0:
            agg_params = fedavg_aggregate(
                cluster_models,
                [s / total_size for s in cluster_sizes]
            )
            global_model_eval = get_model(config).to(device)
            for p, ap in zip(global_model_eval.parameters(), agg_params):
                p.data.copy_(ap)
        else:
            global_model_eval = cluster_models[0]

        # Evaluate: each client uses their assigned cluster model
        global_acc = evaluate_model(global_model_eval, global_test_loader, device)
        client_accs = [
            evaluate_model(cluster_models[client_clusters[i]], client_val_loaders[i], device)
            for i in range(num_clients)
        ]

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs, elapsed)

        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("IFCA", rnd, total_rounds,
                                 global_acc, sum(client_accs)/len(client_accs))

    print(f"  [IFCA] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
