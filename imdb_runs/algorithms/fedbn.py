"""
algorithms/fedbn.py — FedBN baseline.
From: Li et al. 2021, 'FedBN: Federated Learning on Non-IID Features
      via Local Batch Normalization'
Aggregates all parameters EXCEPT BatchNorm — BN layers stay local.
"""

import torch
import copy
import time
from utils import ResultsTracker, print_round_summary, evaluate_model
from local_trainer import local_train_fedbn
from models import get_model, aggregate_except_bn


def run_fedbn(config, client_train_loaders, client_val_loaders,
              global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: FedBN")
    print("="*50)

    tracker = ResultsTracker("FedBN")
    num_clients = config["num_clients"]
    total_rounds = config["total_rounds"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]
    total_data = sum(client_data_sizes)

    global_model = get_model(config).to(device)
    # Each client keeps their own local model (for local BN state)
    local_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()

        # Distribute global non-BN params to all clients
        global_state = global_model.state_dict()
        for i in range(num_clients):
            client_state = local_models[i].state_dict()
            for key in global_state:
                if "bn" not in key:
                    client_state[key] = global_state[key].clone()
            local_models[i].load_state_dict(client_state)

        # Local training
        updated_models = []
        for i in range(num_clients):
            trained = local_train_fedbn(
                local_models[i],
                client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device
            )
            local_models[i] = trained
            updated_models.append(trained)

        # Aggregate non-BN parameters only
        weights = [client_data_sizes[i] / total_data for i in range(num_clients)]
        agg_state = aggregate_except_bn(updated_models, weights)

        # Update global model non-BN params
        global_state = global_model.state_dict()
        global_state.update(agg_state)
        global_model.load_state_dict(global_state)

        # Evaluate: each client uses their own local model (with local BN)
        global_acc = evaluate_model(global_model, global_test_loader, device)
        client_accs = [evaluate_model(local_models[i], client_val_loaders[i], device)
                       for i in range(num_clients)]

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs, elapsed)

        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("FedBN", rnd, total_rounds,
                                 global_acc, sum(client_accs)/len(client_accs))

    print(f"  [FedBN] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
