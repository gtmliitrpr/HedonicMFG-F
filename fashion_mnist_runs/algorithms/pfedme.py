"""
algorithms/pfedme.py — pFedME baseline.
From: Dinh et al. 2020, 'Personalized Federated Learning with Moreau Envelopes'
Each client maintains a personalized model θ_i via Moreau envelope minimization,
while a shared model w is updated via gradient aggregation toward θ_i.
"""

import torch
import copy
import time
import numpy as np
from utils import ResultsTracker, print_round_summary, evaluate_model
from local_trainer import local_train_pfedme
from models import get_model


def run_pfedme(config, client_train_loaders, client_val_loaders,
               global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: pFedME")
    print("="*50)

    tracker = ResultsTracker("pFedME")
    num_clients = config["num_clients"]
    total_rounds = config["total_rounds"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]
    total_data = sum(client_data_sizes)

    global_model = get_model(config).to(device)
    # Personal models per client
    personal_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    beta    = config["pfedme_beta"]
    lam     = config["pfedme_lambda"]
    k_steps = config["pfedme_local_steps"]

    for rnd in range(1, total_rounds + 1):
        t0 = time.time()

        updated_globals = []
        for i in range(num_clients):
            w_i = copy.deepcopy(global_model)
            w_i, personal_models[i] = local_train_pfedme(
                model=w_i,
                global_model=global_model,
                dataloader=client_train_loaders[i],
                local_steps=k_steps,
                lr=config["lr"],
                beta=beta,
                lambda_reg=lam,
                device=device
            )
            updated_globals.append(w_i)

        # Aggregate global model w: w = (1/N) * sum(w_i)
        weights = [client_data_sizes[i] / total_data for i in range(num_clients)]
        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]

        new_state = {}
        ref_state = global_model.state_dict()
        for key in ref_state:
            new_state[key] = sum(
                nw * m.state_dict()[key].float()
                for nw, m in zip(norm_w, updated_globals)
            )
        global_model.load_state_dict(new_state)

        # Evaluate using personalized models per client
        global_acc = evaluate_model(global_model, global_test_loader, device)
        client_accs = [evaluate_model(personal_models[i], client_val_loaders[i], device)
                       for i in range(num_clients)]

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs, elapsed)

        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("pFedME", rnd, total_rounds,
                                 global_acc, sum(client_accs)/len(client_accs))

    print(f"  [pFedME] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
