"""
algorithms/hedonic_mfg.py — HedonicMFG (Full Improved Version, CIFAR-10 ready)

Key improvements over original:
  1. Real gradient-based similarity matrix for coalition formation
  2. Adaptive reclustering triggered by within-coalition similarity drop
  3. ResNet-style backbone (512-dim) for CIFAR-10
  4. Cosine-annealing LR scheduler inside local training for CIFAR-10
  5. Personalized head finetuned with lr*2 (aggressive personalisation)
  6. Warmup = 20 rounds for CIFAR-10 (gradients more informative before coalitions)
  7. K=4 coalitions for N=20 (~5 clients each — tighter, more homogeneous groups)
  8. Nash iterations = 10 for more stable equilibrium
  9. Broader MFG epoch space [3,5,7,10,15]
 10. Coalition models inherit dominant-previous-coalition model on recluster
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import numpy as np
from utils import (ResultsTracker, print_round_summary, evaluate_model,
                    evaluate_with_head, fedavg_aggregate,
                    pairwise_cosine_similarity)
from models import get_model, get_personal_head


# ═══════════════════════════════════════════════════════════
# GRADIENT SIMILARITY  — real per-client gradient vectors
# ═══════════════════════════════════════════════════════════

def compute_grad_sim_matrix(global_model, client_train_loaders,
                             num_clients, device):
    """
    One forward+backward pass per client on one mini-batch.
    Returns N×N pairwise cosine similarity matrix of gradient vectors.
    This is the correct hedonic similarity signal.
    """
    criterion = nn.CrossEntropyLoss()
    grad_vecs  = []

    for i in range(num_clients):
        tmp = copy.deepcopy(global_model).to(device)
        tmp.train()
        for x, y in client_train_loaders[i]:
            x, y = x.to(device), y.to(device)
            tmp.zero_grad()
            loss = criterion(tmp(x), y)
            loss.backward()
            break   # one batch is sufficient for direction

        grads = torch.cat([
            p.grad.data.clone().flatten()
            for p in tmp.parameters() if p.grad is not None
        ])
        grad_vecs.append(grads)
        del tmp

    return pairwise_cosine_similarity(grad_vecs)


# ═══════════════════════════════════════════════════════════
# HEDONIC GAME — Nash-stable coalition formation
# ═══════════════════════════════════════════════════════════

def compute_hedonic_utility(client_id, coalition, client_accs,
                             grad_sim_matrix, config):
    """
    phi_i(S) = lambda_perf * Acc_i
             - beta_size * |S|
             + gamma_grad * avg_cosine_sim(i, S minus i)
             + mu_fair * min_{j in S} Acc_j
    """
    if not coalition:
        return -float("inf")

    λ = config["lambda_perf"]
    β = config["beta_size"]
    γ = config["gamma_grad"]
    μ = config["mu_fair"]

    perf     = λ * client_accs.get(client_id, 0.0)
    size_pen = β * len(coalition)
    sims     = [grad_sim_matrix[client_id][j]
                for j in coalition if j != client_id]
    grad_sim = γ * (np.mean(sims) if sims else 0.0)
    fairness = μ * min(client_accs.get(j, 0.0) for j in coalition)

    return perf - size_pen + grad_sim + fairness


def form_coalitions_hedonic(num_clients, K, client_accs, grad_sim_matrix,
                             config, prev_coalitions=None):
    """Nash-stable coalition formation."""
    min_size   = config["min_coalition_size"]
    nash_iters = config["nash_iterations"]
    rng        = np.random.RandomState(config.get("round_seed", 42))

    if prev_coalitions is not None:
        c2k = {c: k for k, coal in enumerate(prev_coalitions) for c in coal}
    else:
        shuffled = list(range(num_clients))
        rng.shuffle(shuffled)
        c2k = {c: shuffled.index(c) % K for c in range(num_clients)}

    for _ in range(nash_iters):
        improved = False
        order = list(range(num_clients))
        rng.shuffle(order)
        for client in order:
            curr_k    = c2k[client]
            curr_coal = [c for c, k in c2k.items() if k == curr_k]
            curr_u    = compute_hedonic_utility(client, curr_coal,
                                                client_accs, grad_sim_matrix, config)
            best_k, best_u = curr_k, curr_u
            for tgt_k in range(K):
                if tgt_k == curr_k:
                    continue
                src_after = [c for c in curr_coal if c != client]
                if 0 < len(src_after) < min_size:
                    continue
                tgt_coal = [c for c, k in c2k.items() if k == tgt_k] + [client]
                u = compute_hedonic_utility(client, tgt_coal,
                                            client_accs, grad_sim_matrix, config)
                if u > best_u + 1e-6:
                    best_u, best_k = u, tgt_k
            if best_k != curr_k:
                c2k[client] = best_k
                improved = True
        if not improved:
            break

    coalitions = [[] for _ in range(K)]
    for c, k in c2k.items():
        coalitions[k].append(c)

    # Enforce min_coalition_size
    final, small = [], []
    for coal in coalitions:
        if len(coal) >= min_size:
            final.append(coal)
        else:
            small.extend(coal)
    if small:
        if final:
            final[0].extend(small)
        else:
            final = [small]
    while len(final) < K and any(len(c) > min_size * 2 for c in final):
        for i, coal in enumerate(final):
            if len(coal) > min_size * 2:
                mid = len(coal) // 2
                final[i] = coal[:mid]
                final.append(coal[mid:])
                break
    return final


# ═══════════════════════════════════════════════════════════
# MFG — Mean-Field Game action optimisation
# ═══════════════════════════════════════════════════════════

def compute_mfg_payoff(action, mean_field_action, base_acc,
                        coalition_min_acc, config):
    epochs, participation = action
    mean_epochs, mean_part = mean_field_action

    effort   = (epochs / 15.0) * participation
    est_acc  = min(1.0, base_acc * (1.0 + 0.08 * effort))

    comp_cost  = config["alpha_comp"]    * epochs
    part_cost  = config["beta_part"]     * (1.0 - participation)
    sync_pen   = config["gamma_sync"]    * np.linalg.norm(
                     np.array([epochs, participation]) -
                     np.array([mean_epochs, mean_part])) ** 2
    fairness   = config["lambda_fair_mfg"] * coalition_min_acc
    contrib    = config["delta_contrib"]   * (epochs / 15.0)

    return est_acc - comp_cost - part_cost - sync_pen + fairness + contrib


def solve_mfg_fast(coalition, client_accs, config):
    """Fast MFG with coarse grid — 5 epoch choices × 3 participation = 15 actions."""
    epoch_space   = [3, 5, 7, 10, 15]
    part_space    = [0.85, 0.9, 1.0]
    max_iter      = config["mfg_iterations"]

    client_actions    = {c: (config["local_epochs_base"], 1.0) for c in coalition}
    accs              = [client_accs.get(c, 0.5) for c in coalition]
    coalition_min_acc = min(accs) if accs else 0.5

    for _ in range(max_iter):
        mean_epochs = np.mean([a[0] for a in client_actions.values()])
        mean_part   = np.mean([a[1] for a in client_actions.values()])
        mean_field  = (mean_epochs, mean_part)
        old_actions = client_actions.copy()

        for c in coalition:
            base_acc  = client_accs.get(c, 0.5)
            best_pay  = -float('inf')
            best_act  = client_actions[c]
            for e in epoch_space:
                for tau in part_space:
                    pay = compute_mfg_payoff(
                        (e, tau), mean_field, base_acc,
                        coalition_min_acc, config)
                    if pay > best_pay:
                        best_pay, best_act = pay, (e, tau)
            client_actions[c] = best_act

        changes = [abs(client_actions[c][0] - old_actions[c][0]) +
                   abs(client_actions[c][1] - old_actions[c][1])
                   for c in coalition]
        if max(changes) < 0.5:
            break

    return client_actions


# ═══════════════════════════════════════════════════════════
# LOCAL TRAINING  (with cosine-annealing LR for CIFAR-10)
# ═══════════════════════════════════════════════════════════

def local_train_client(model, dataloader, epochs, lr, momentum,
                        weight_decay, device,
                        personal_head=None, finetune_epochs=3,
                        use_scheduler=False):
    """
    Local SGD with optional cosine-annealing LR scheduler.
    use_scheduler=True is set for CIFAR-10 to counter client drift.
    Personal head finetuned with lr*2 (aggressive personalisation).
    """
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                           momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Cosine annealing over local epochs — smoothly reduces LR inside each round
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if use_scheduler else None)

    for ep in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # Fine-tune personalized head — aggressive lr*2
    if personal_head is not None:
        model.eval()
        personal_head = personal_head.to(device)
        personal_head.train()
        head_opt = optim.Adam(personal_head.parameters(), lr=lr * 2,
                               weight_decay=1e-5)
        for _ in range(finetune_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                head_opt.zero_grad()
                with torch.no_grad():
                    feat = model.get_features(x)
                criterion(personal_head(feat), y).backward()
                head_opt.step()

    return model, personal_head


# ═══════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════

def run_hedonic_mfg(config, client_train_loaders, client_val_loaders,
                    global_test_loader, device):
    dataset = config.get("dataset", "mnist")
    use_scheduler = (dataset == "cifar10")   # cosine annealing only for CIFAR-10

    print("\n" + "="*50)
    print("  Running: HedonicMFG (Improved)")
    print("="*50)

    tracker            = ResultsTracker("HedonicMFG")
    num_clients        = config["num_clients"]
    total_rounds       = config["total_rounds"]
    K                  = config["num_coalitions"]
    warmup_rounds      = config["warmup_rounds"]
    recluster_interval = config["recluster_interval"]
    client_data_sizes  = [len(l.dataset) for l in client_train_loaders]
    total_data         = sum(client_data_sizes)
    use_ph             = config.get("use_personalized_head", True)

    # ── Phase 1: FedAvg Warmup ──────────────────────────
    print(f"\n  [Phase 1] Warmup ({warmup_rounds} rounds) ...")
    global_model  = get_model(config).to(device)
    client_models = {i: copy.deepcopy(global_model) for i in range(num_clients)}

    for rnd in range(1, warmup_rounds + 1):
        local_models = []
        for i in range(num_clients):
            lm = copy.deepcopy(global_model)
            lm, _ = local_train_client(
                lm, client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"], momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device, use_scheduler=use_scheduler)
            client_models[i] = lm
            local_models.append(lm)

        weights    = [s / total_data for s in client_data_sizes]
        agg_params = fedavg_aggregate(local_models, weights)
        for p, ap in zip(global_model.parameters(), agg_params):
            p.data.copy_(ap)

    warmup_acc = evaluate_model(global_model, global_test_loader, device)
    print(f"  [Phase 1] Warmup complete. Global acc: {warmup_acc*100:.2f}%")

    # ── Phase 2: Initial coalition formation ─────────────
    print(f"\n  [Phase 2] Initial coalition formation (K={K})...")

    client_accs_dict = {
        i: evaluate_model(client_models[i], client_val_loaders[i], device)
        for i in range(num_clients)
    }

    # Real gradient similarity matrix
    grad_sim_matrix = compute_grad_sim_matrix(
        global_model, client_train_loaders, num_clients, device)

    config["round_seed"] = config["seed"]
    coalitions = form_coalitions_hedonic(
        num_clients, K, client_accs_dict, grad_sim_matrix, config)
    print(f"  Coalitions: {coalitions}")

    # Coalition models = weighted aggregate of member client models
    c2k = {c: k for k, coal in enumerate(coalitions) for c in coal}
    coalition_models = {}
    for k, coal in enumerate(coalitions):
        members    = [client_models[c] for c in coal]
        coal_sizes = [client_data_sizes[c] for c in coal]
        tw         = sum(coal_sizes)
        agg        = fedavg_aggregate(members, [s/tw for s in coal_sizes])
        coalition_models[k] = copy.deepcopy(global_model)
        for p, ap in zip(coalition_models[k].parameters(), agg):
            p.data.copy_(ap)

    personal_heads = {i: get_personal_head(config).to(device)
                      for i in range(num_clients)} if use_ph else {}

    last_sim_score = 1.0

    # ── Phase 3: Clustered training with MFG ─────────────
    print(f"\n  [Phase 3] Clustered MFG training "
          f"({total_rounds - warmup_rounds} rounds)...")

    for rnd in range(warmup_rounds + 1, total_rounds + 1):
        t0 = time.time()
        config["round_seed"] = config["seed"] + rnd

        # Adaptive reclustering
        if (rnd - warmup_rounds) % recluster_interval == 0:
            grad_sim_matrix = compute_grad_sim_matrix(
                global_model, client_train_loaders, num_clients, device)

            client_accs_dict = {
                i: evaluate_model(coalition_models[c2k[i]],
                                   client_val_loaders[i], device)
                for i in range(num_clients)
            }

            within_sims = [
                grad_sim_matrix[ci][cj]
                for k, coal in enumerate(coalitions)
                for idx, ci in enumerate(coal)
                for cj in coal[idx+1:]
            ]
            avg_within_sim = np.mean(within_sims) if within_sims else 1.0

            if avg_within_sim < last_sim_score - 0.05 or avg_within_sim < 0.3:
                old_coalitions = coalitions
                coalitions = form_coalitions_hedonic(
                    num_clients, K, client_accs_dict,
                    grad_sim_matrix, config,
                    prev_coalitions=old_coalitions)

                new_c2k = {c: k for k, coal in enumerate(coalitions)
                           for c in coal}
                new_cmodels = {}
                for k, coal in enumerate(coalitions):
                    prev_ks = [c2k.get(c, 0) for c in coal]
                    dom_k   = max(set(prev_ks), key=prev_ks.count)
                    src_k   = dom_k if dom_k in coalition_models else 0
                    new_cmodels[k] = copy.deepcopy(coalition_models[src_k])

                coalition_models = new_cmodels
                c2k              = new_c2k
                last_sim_score   = avg_within_sim
                print(f"  [Recluster] Round {rnd} — "
                      f"within-sim: {avg_within_sim:.3f} → {coalitions}")

        # Solve MFG per coalition
        all_strats = {}
        for k, coal in enumerate(coalitions):
            coal_accs = {c: client_accs_dict.get(c, 0.5) for c in coal}
            all_strats.update(solve_mfg_fast(coal, coal_accs, config))

        # Local training
        cu = {k: [] for k in range(len(coalitions))}
        cw = {k: [] for k in range(len(coalitions))}

        for i in range(num_clients):
            k   = c2k[i]
            lm  = copy.deepcopy(coalition_models[k])
            opt_e, _ = all_strats.get(i, (config["local_epochs_base"], 1.0))
            ph  = personal_heads.get(i) if use_ph else None

            trained, trained_head = local_train_client(
                lm, client_train_loaders[i],
                epochs=opt_e,
                lr=config["lr"], momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device,
                personal_head=copy.deepcopy(ph) if ph is not None else None,
                finetune_epochs=config.get("finetune_rounds", 3),
                use_scheduler=use_scheduler)

            client_models[i] = trained
            cu[k].append(trained)
            cw[k].append(client_data_sizes[i])
            if trained_head is not None:
                personal_heads[i] = trained_head

        # Coalition-level aggregation
        for k in range(len(coalitions)):
            if cu[k]:
                tw  = sum(cw[k])
                agg = fedavg_aggregate(cu[k], [w/tw for w in cw[k]])
                for p, ap in zip(coalition_models[k].parameters(), agg):
                    p.data.copy_(ap)

        # Meta-aggregation → global model for evaluation
        coal_sizes = [sum(client_data_sizes[c] for c in coal)
                      for coal in coalitions]
        total_coal = sum(coal_sizes)
        valid_ks   = [k for k, s in enumerate(coal_sizes)
                      if s > 0 and k in coalition_models]

        global_model_eval = copy.deepcopy(coalition_models[valid_ks[0]])
        if len(valid_ks) > 1:
            agg = fedavg_aggregate(
                [coalition_models[k] for k in valid_ks],
                [coal_sizes[k]/total_coal for k in valid_ks])
            for p, ap in zip(global_model_eval.parameters(), agg):
                p.data.copy_(ap)

        global_model = global_model_eval

        # Evaluate
        global_acc  = evaluate_model(global_model_eval, global_test_loader, device)
        c_accs_list = []
        for i in range(num_clients):
            k = c2k[i]
            if use_ph and i in personal_heads:
                acc = evaluate_with_head(coalition_models[k], personal_heads[i],
                                          client_val_loaders[i], device)
            else:
                acc = evaluate_model(client_models[i],
                                      client_val_loaders[i], device)
            c_accs_list.append(acc)

        client_accs_dict = {i: c_accs_list[i] for i in range(num_clients)}

        tracker.log(rnd, global_acc, c_accs_list, time.time() - t0)
        if rnd % 10 == 0 or rnd == warmup_rounds + 1:
            print_round_summary("HedonicMFG", rnd, total_rounds,
                                 global_acc,
                                 sum(c_accs_list)/len(c_accs_list))

    print(f"\n  [HedonicMFG] Best global: "
          f"{max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
