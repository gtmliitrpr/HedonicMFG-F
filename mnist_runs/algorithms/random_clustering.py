"""
algorithms/random_clustering.py — Random Clustering baseline.
Pure random assignment of clients to K clusters. No gradient-based logic.
Clusters are randomly reshuffled every R rounds (same interval as HedonicMFG).
This is the ablation that proves HedonicMFG's gradient-similarity clustering
does real work vs. pure random grouping.
"""

import torch
import copy
import time
import numpy as np
import random
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from local_trainer import local_train_standard
from models import get_model


def random_partition(num_clients: int, K: int, seed: int = 42) -> list:
    """Randomly assign clients to K clusters. Returns list of lists."""
    rng = np.random.RandomState(seed)
    clients = list(range(num_clients))
    rng.shuffle(clients)
    clusters = [[] for _ in range(K)]
    for idx, client_id in enumerate(clients):
        clusters[idx % K].append(client_id)
    return [c for c in clusters if c]  # remove empty


def run_random_clustering(config, client_train_loaders, client_val_loaders,
                           global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: Random Clustering")
    print("="*50)

    tracker = ResultsTracker("RandomCluster")
    num_clients = config["num_clients"]
    total_rounds = config["total_rounds"]
    K = config["random_clustering_K"]
    R = config["recluster_interval"]          # Recluster every R rounds
    warmup = config["warmup_rounds"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]

    # Phase 1: Warmup with FedAvg
    global_model = get_model(config).to(device)
    total_data = sum(client_data_sizes)

    print(f"  [RandomCluster] Warmup phase: {warmup} rounds")
    for rnd in range(1, warmup + 1):
        local_models = []
        for i in range(num_clients):
            lm = copy.deepcopy(global_model)
            lm = local_train_standard(
                lm, client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"], momentum=config["momentum"],
                weight_decay=config["weight_decay"], device=device
            )
            local_models.append(lm)
        weights = [s / total_data for s in client_data_sizes]
        agg_params = fedavg_aggregate(local_models, weights)
        for p, ap in zip(global_model.parameters(), agg_params):
            p.data.copy_(ap)

    # Phase 2: Clustered training with random assignment
    clusters = random_partition(num_clients, K, seed=config["seed"])
    cluster_models = {k: copy.deepcopy(global_model) for k in range(len(clusters))}
    client_to_cluster = {}
    for k, c in enumerate(clusters):
        for cid in c:
            client_to_cluster[cid] = k

    print(f"  [RandomCluster] Initial clusters: {clusters}")

    for rnd in range(warmup + 1, total_rounds + 1):
        t0 = time.time()

        # Rerandomize every R rounds
        if (rnd - warmup) % R == 0:
            clusters = random_partition(num_clients, K,
                                         seed=config["seed"] + rnd)
            # Re-use existing cluster models but reassign clients
            num_clusters = len(clusters)
            for k, c in enumerate(clusters):
                for cid in c:
                    client_to_cluster[cid] = k % len(cluster_models)

        # Train within each cluster
        cluster_updates = {k: [] for k in range(len(cluster_models))}
        cluster_w = {k: [] for k in range(len(cluster_models))}

        for i in range(num_clients):
            k = client_to_cluster[i]
            lm = copy.deepcopy(cluster_models[k])
            lm = local_train_standard(
                lm, client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"], momentum=config["momentum"],
                weight_decay=config["weight_decay"], device=device
            )
            cluster_updates[k].append(lm)
            cluster_w[k].append(client_data_sizes[i])

        # Aggregate within clusters
        for k in cluster_updates:
            if cluster_updates[k]:
                tw = sum(cluster_w[k])
                nw = [w / tw for w in cluster_w[k]]
                agg_params = fedavg_aggregate(cluster_updates[k], nw)
                for p, ap in zip(cluster_models[k].parameters(), agg_params):
                    p.data.copy_(ap)

        # Global model via meta-aggregation
        cluster_sizes = [sum(client_data_sizes[i] for i in range(num_clients)
                             if client_to_cluster.get(i) == k)
                         for k in range(len(cluster_models))]
        total_size = sum(s for s in cluster_sizes if s > 0)
        valid_models = [cluster_models[k] for k, s in enumerate(cluster_sizes) if s > 0]
        valid_weights = [s / total_size for s in cluster_sizes if s > 0]

        if valid_models:
            agg_params = fedavg_aggregate(valid_models, valid_weights)
            for p, ap in zip(global_model.parameters(), agg_params):
                p.data.copy_(ap)

        global_acc = evaluate_model(global_model, global_test_loader, device)
        client_accs = [
            evaluate_model(cluster_models[client_to_cluster[i]],
                           client_val_loaders[i], device)
            for i in range(num_clients)
        ]

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs, elapsed)

        if rnd % 10 == 0 or rnd == warmup + 1:
            print_round_summary("RandomCluster", rnd, total_rounds,
                                 global_acc, sum(client_accs)/len(client_accs))

    print(f"  [RandomCluster] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
