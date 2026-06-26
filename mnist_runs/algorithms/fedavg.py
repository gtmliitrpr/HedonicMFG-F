"""
algorithms/fedavg.py — FedAvg baseline.
From: McMahan et al. 2017, 'Communication-Efficient Learning of Deep Networks
      from Decentralized Data'
"""

import torch
import copy
import time
from utils import (ResultsTracker, print_round_summary,
                    evaluate_model, fedavg_aggregate)
from local_trainer import local_train_standard
from models import get_model


def run_fedavg(config, client_train_loaders, client_val_loaders,
               global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: FedAvg")
    print("="*50)

    tracker = ResultsTracker("FedAvg")
    num_clients = config["num_clients"]
    total_rounds = config["total_rounds"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]
    total_data = sum(client_data_sizes)

    # Initialize global model
    global_model = get_model(config).to(device)

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()

        # All clients participate
        local_models = []
        for i in range(num_clients):
            local_model = copy.deepcopy(global_model)
            local_model = local_train_standard(
                local_model, client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device
            )
            local_models.append(local_model)

        # Aggregate
        weights = [client_data_sizes[i] / total_data for i in range(num_clients)]
        agg_params = fedavg_aggregate(local_models, weights)
        for p, ap in zip(global_model.parameters(), agg_params):
            p.data.copy_(ap)

        # Evaluate
        global_acc = evaluate_model(global_model, global_test_loader, device)
        client_accs = [evaluate_model(global_model, client_val_loaders[i], device)
                       for i in range(num_clients)]

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs, elapsed)

        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("FedAvg", rnd, total_rounds,
                                 global_acc, sum(client_accs)/len(client_accs))

    print(f"  [FedAvg] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
