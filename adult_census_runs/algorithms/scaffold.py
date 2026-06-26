"""
algorithms/scaffold.py — SCAFFOLD baseline.
From: Karimireddy et al. 2020, 'SCAFFOLD: Stochastic Controlled Averaging
      for Federated Learning'
Corrects client drift using control variates c_i and c (global).
"""

import torch
import copy
import time
from utils import ResultsTracker, print_round_summary, evaluate_model
from local_trainer import local_train_scaffold
from models import get_model


def run_scaffold(config, client_train_loaders, client_val_loaders,
                 global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: SCAFFOLD")
    print("="*50)

    tracker = ResultsTracker("SCAFFOLD")
    num_clients = config["num_clients"]
    total_rounds = config["total_rounds"]
    lr = config["scaffold_lr"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]
    total_data = sum(client_data_sizes)

    global_model = get_model(config).to(device)

    # Initialize control variates to zero
    c_global = [torch.zeros_like(p) for p in global_model.parameters()]
    c_clients = [[torch.zeros_like(p) for p in global_model.parameters()]
                 for _ in range(num_clients)]

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()

        local_models = []
        c_deltas = []

        for i in range(num_clients):
            local_model = copy.deepcopy(global_model)
            local_model, new_c_i, c_delta = local_train_scaffold(
                local_model, global_model,
                client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=lr,
                c_i=c_clients[i],
                c_global=c_global,
                device=device
            )
            c_clients[i] = new_c_i
            local_models.append(local_model)
            c_deltas.append(c_delta)

        # Aggregate global model
        weights = [client_data_sizes[i] / total_data for i in range(num_clients)]
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        for p_global, *p_locals_list in zip(global_model.parameters(),
                                             *[m.parameters() for m in local_models]):
            p_global.data = sum(
                w * p.data for w, p in zip(norm_weights, p_locals_list)
            )

        # Update global control variate: c += (1/N) * sum(c_delta)
        for c_g, *deltas in zip(c_global, *c_deltas):
            c_g.data += sum(d for d in deltas) / num_clients

        global_acc = evaluate_model(global_model, global_test_loader, device)
        client_accs = [evaluate_model(global_model, client_val_loaders[i], device)
                       for i in range(num_clients)]

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs, elapsed)

        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("SCAFFOLD", rnd, total_rounds,
                                 global_acc, sum(client_accs)/len(client_accs))

    print(f"  [SCAFFOLD] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
