"""
algorithms/cfl.py — CFL (Clustered Federated Learning) baseline.
From: Sattler et al. 2020, 'Clustered Federated Learning: Model-Agnostic
      Distributed Multitask Optimization Under Privacy Constraints'
Clusters clients by cosine similarity of gradient updates.
Splits clusters when bipartite structure is detected (eps1, eps2 thresholds).
"""

import torch
import copy
import time
import numpy as np
from utils import (ResultsTracker, print_round_summary, evaluate_model,
                    fedavg_aggregate, get_model_params, pairwise_cosine_similarity)
from local_trainer import local_train_standard
from models import get_model


def run_cfl(config, client_train_loaders, client_val_loaders,
            global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: CFL")
    print("="*50)

    tracker = ResultsTracker("CFL")
    num_clients = config["num_clients"]
    total_rounds = config["total_rounds"]
    eps1 = config["cfl_eps1"]   # max mean cosine similarity to trigger split
    eps2 = config["cfl_eps2"]   # min cosine similarity norm for split
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]

    # Initial: all clients in one cluster
    clusters = [list(range(num_clients))]
    cluster_models = {0: get_model(config).to(device)}
    client_to_cluster = {i: 0 for i in range(num_clients)}
    next_cluster_id = 1

    def compute_updates(cluster_id, client_ids):
        """Train and return gradient updates for all clients in cluster."""
        model = cluster_models[cluster_id]
        updates = {}
        trained = {}
        init_params = get_model_params(model)
        for i in client_ids:
            local_model = copy.deepcopy(model)
            local_model = local_train_standard(
                local_model, client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device
            )
            trained[i] = local_model
            # Update = new_params - old_params (flattened)
            local_params = get_model_params(local_model)
            diff = torch.cat([(lp - ip).flatten()
                               for lp, ip in zip(local_params, init_params)])
            updates[i] = diff
        return updates, trained

    def try_split(cluster_id, client_ids, updates):
        """Check bipartite condition and split if needed."""
        nonlocal next_cluster_id
        if len(client_ids) < 2:
            return False

        vecs = [updates[i] for i in client_ids]
        sim_matrix = pairwise_cosine_similarity(vecs)
        mean_sim = sim_matrix.mean()
        norms = [v.norm().item() for v in vecs]
        max_norm_diff = max(norms) - min(norms)

        # CFL split condition: low mean cosine sim AND large norm difference
        if mean_sim < eps1 and max_norm_diff > eps2:
            # Split: clients with above-mean norm vs. below-mean norm
            mean_norm = np.mean(norms)
            group_a = [cid for cid, n in zip(client_ids, norms) if n >= mean_norm]
            group_b = [cid for cid, n in zip(client_ids, norms) if n < mean_norm]

            if not group_a or not group_b:
                return False

            # Create new cluster for group_b
            new_id = next_cluster_id
            next_cluster_id += 1
            cluster_models[new_id] = copy.deepcopy(cluster_models[cluster_id])

            # Update cluster lists
            clusters.remove(client_ids)
            clusters.append(group_a)
            clusters.append(group_b)

            for c in group_a:
                client_to_cluster[c] = cluster_id
            for c in group_b:
                client_to_cluster[c] = new_id

            print(f"    [CFL] Round cluster split: {cluster_id}→{group_a}, "
                  f"{new_id}→{group_b}")
            return True
        return False

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()

        all_trained = {}
        all_updates = {}

        # Train within each cluster
        for cluster_clients in list(clusters):
            cluster_id = client_to_cluster[cluster_clients[0]]
            updates, trained = compute_updates(cluster_id, cluster_clients)
            all_updates.update(updates)
            all_trained.update(trained)

            # Try split
            if rnd > 5:  # Only attempt split after warmup
                try_split(cluster_id, cluster_clients, updates)

        # Aggregate within each cluster
        for cluster_clients in list(clusters):
            cluster_id = client_to_cluster[cluster_clients[0]]
            models_in_cluster = [all_trained[i] for i in cluster_clients if i in all_trained]
            weights = [client_data_sizes[i] for i in cluster_clients if i in all_trained]

            if models_in_cluster:
                total_w = sum(weights)
                norm_w = [w / total_w for w in weights]
                agg_params = fedavg_aggregate(models_in_cluster, norm_w)
                for p, ap in zip(cluster_models[cluster_id].parameters(), agg_params):
                    p.data.copy_(ap)

        # Global model = weighted average across all cluster models
        all_cluster_ids = list({client_to_cluster[i] for i in range(num_clients)})
        cluster_pop = {cid: sum(client_data_sizes[i] for i in range(num_clients)
                                if client_to_cluster[i] == cid)
                       for cid in all_cluster_ids}
        total_pop = sum(cluster_pop.values())

        global_model_eval = get_model(config).to(device)
        agg_params = fedavg_aggregate(
            [cluster_models[cid] for cid in all_cluster_ids],
            [cluster_pop[cid] / total_pop for cid in all_cluster_ids]
        )
        for p, ap in zip(global_model_eval.parameters(), agg_params):
            p.data.copy_(ap)

        global_acc = evaluate_model(global_model_eval, global_test_loader, device)
        client_accs = [
            evaluate_model(cluster_models[client_to_cluster[i]],
                           client_val_loaders[i], device)
            for i in range(num_clients)
        ]

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs, elapsed)

        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("CFL", rnd, total_rounds,
                                 global_acc, sum(client_accs)/len(client_accs))

    print(f"  [CFL] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
