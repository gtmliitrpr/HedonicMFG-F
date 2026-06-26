"""
algorithms/scaffold.py — SCAFFOLD baseline.
From: Karimireddy et al. 2020, 'SCAFFOLD: Stochastic Controlled Averaging
      for Federated Learning'

Fix: all control variates (c_global, c_clients, c_delta) kept on CPU.
     Moved to device only inside training for gradient correction,
     then immediately returned to CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from models import get_model


def local_train_scaffold_fixed(model, global_model, dataloader, epochs,
                                lr, c_i, c_global, device):
    """
    SCAFFOLD local training.
    c_i, c_global : CPU tensors — moved to device only for correction step.
    Returns       : (model, new_c_i_cpu, c_delta_cpu)
    """
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    # Move to device only for the correction computation
    c_i_dev = [c.to(device) for c in c_i]
    c_g_dev = [c.to(device) for c in c_global]

    init_params = [p.data.clone() for p in model.parameters()]
    num_steps = 0

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            # SCAFFOLD correction: subtract c_i, add c_global
            for p, ci, cg in zip(model.parameters(), c_i_dev, c_g_dev):
                if p.grad is not None:
                    p.grad.data.add_(cg - ci)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            num_steps += 1

    if num_steps == 0:
        num_steps = 1

    # Compute updated c_i and delta — return as CPU tensors
    new_c_i = []
    c_delta  = []
    for ci_d, cg_d, ip, p in zip(c_i_dev, c_g_dev, init_params, model.parameters()):
        new_ci = ci_d - cg_d + (ip.to(device) - p.data) / (num_steps * lr)
        new_c_i.append(new_ci.cpu())          # back to CPU
        c_delta.append((new_ci - ci_d).cpu()) # back to CPU

    return model, new_c_i, c_delta


def run_scaffold(config, client_train_loaders, client_val_loaders,
                 global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: SCAFFOLD")
    print("="*50)

    tracker           = ResultsTracker("SCAFFOLD")
    num_clients       = config["num_clients"]
    total_rounds      = config["total_rounds"]
    lr                = config["scaffold_lr"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]
    total_data        = sum(client_data_sizes)

    global_model = get_model(config).to(device)

    # Initialise ALL control variates explicitly on CPU
    c_global  = [torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                 for p in global_model.parameters()]
    c_clients = [[torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                  for p in global_model.parameters()]
                 for _ in range(num_clients)]

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()

        local_models = []
        c_deltas     = []

        for i in range(num_clients):
            local_model = copy.deepcopy(global_model)
            local_model, new_c_i, c_delta = local_train_scaffold_fixed(
                local_model, global_model,
                client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=lr,
                c_i=c_clients[i],    # CPU
                c_global=c_global,   # CPU
                device=device
            )
            c_clients[i] = new_c_i   # store CPU tensors
            local_models.append(local_model)
            c_deltas.append(c_delta)  # CPU tensors

        # Aggregate global model
        weights    = [client_data_sizes[i] / total_data for i in range(num_clients)]
        agg_params = fedavg_aggregate(local_models, weights)
        for p, ap in zip(global_model.parameters(), agg_params):
            p.data.copy_(ap.to(device))

        # Update global control variate — everything on CPU
        for idx, c_g in enumerate(c_global):
            delta_sum = sum(c_deltas[i][idx] for i in range(num_clients))
            c_g.data.add_(delta_sum / num_clients)  # CPU + CPU — no device mismatch

        global_acc  = evaluate_model(global_model, global_test_loader, device)
        client_accs = [evaluate_model(global_model, client_val_loaders[i], device)
                       for i in range(num_clients)]

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs, elapsed)

        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("SCAFFOLD", rnd, total_rounds,
                                 global_acc, sum(client_accs) / len(client_accs))

    print(f"  [SCAFFOLD] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
