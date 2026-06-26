"""
SCAFFOLD — Adult Census.
Fix: control variates initialised explicitly on CPU, moved to device
only during training, returned to CPU immediately after.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy, time
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from models_adult import get_adult_model


def local_train_adult_scaffold_fixed(model, global_model, dataloader, epochs,
                                      lr, c_i, c_global, device):
    """
    SCAFFOLD local training — c_i and c_global are CPU tensors.
    Moved to device only for gradient correction, returned as CPU.
    """
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    # Move control variates to device temporarily
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
            for p, ci, cg in zip(model.parameters(), c_i_dev, c_g_dev):
                if p.grad is not None:
                    p.grad.data.add_(cg - ci)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            num_steps += 1

    if num_steps == 0:
        num_steps = 1

    # Compute new c_i and delta — return as CPU tensors
    new_c_i, c_delta = [], []
    for ci_d, cg_d, ip, p in zip(c_i_dev, c_g_dev, init_params, model.parameters()):
        new_ci = ci_d - cg_d + (ip.to(device) - p.data) / (num_steps * lr)
        new_c_i.append(new_ci.cpu())
        c_delta.append((new_ci - ci_d).cpu())

    return model, new_c_i, c_delta


def run_scaffold_adult(config, client_train_loaders, client_val_loaders,
                       global_test_loader, device, feature_dim):
    print("\n" + "="*50 + "\n  Running: SCAFFOLD [Adult]\n" + "="*50)
    tracker = ResultsTracker("SCAFFOLD")
    N     = config["num_clients"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    total = sum(sizes)
    lr    = config["scaffold_lr"]

    global_model = get_adult_model(config, feature_dim).to(device)

    # ALL control variates explicitly on CPU
    c_global  = [torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                 for p in global_model.parameters()]
    c_clients = [[torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                  for p in global_model.parameters()]
                 for _ in range(N)]

    for rnd in range(1, config["total_rounds"] + 1):
        t0 = time.time()
        local_models, c_deltas = [], []

        for i in range(N):
            lm = copy.deepcopy(global_model)
            lm, new_ci, c_delta = local_train_adult_scaffold_fixed(
                lm, global_model, client_train_loaders[i],
                config["local_epochs_base"], lr,
                c_clients[i], c_global, device)
            c_clients[i] = new_ci   # CPU
            local_models.append(lm)
            c_deltas.append(c_delta)  # CPU

        # Aggregate global model
        weights    = [s / total for s in sizes]
        agg_params = fedavg_aggregate(local_models, weights)
        for p, ap in zip(global_model.parameters(), agg_params):
            p.data.copy_(ap.to(device))

        # Update global control variate — pure CPU arithmetic
        for idx, c_g in enumerate(c_global):
            delta_sum = sum(c_deltas[i][idx] for i in range(N))
            c_g.data.add_(delta_sum / N)

        g_acc  = evaluate_model(global_model, global_test_loader, device)
        c_accs = [evaluate_model(global_model, client_val_loaders[i], device)
                  for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time() - t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("SCAFFOLD", rnd, config["total_rounds"],
                                 g_acc, sum(c_accs) / N)

    print(f"  [SCAFFOLD] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
