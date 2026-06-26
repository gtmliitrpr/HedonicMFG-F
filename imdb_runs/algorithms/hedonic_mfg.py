"""
algorithms/hedonic_mfg.py — HedonicMFG integrated from working implementation.

Core logic ported from user's working hedonic_mfg.py:
- Sinkhorn distance for hedonic utility (from working version)
- Fast MFG with coarse grid (from working version)
- Coalition model = aggregate of member models (not copy of global)
- Client models tracked individually (not just coalition models)
- Personal head evaluation kept from our pipeline
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
# SINKHORN DISTANCE (from working implementation)
# ═══════════════════════════════════════════════════════════

def compute_gradient_distance(weights_i, weights_s, reg=0.1):
    """
    Compute distance between two weight vectors using cosine distance.
    Simplified from Sinkhorn — works without POT library.
    """
    flat_i = torch.cat([w.flatten().float() for w in weights_i])
    flat_s = torch.cat([w.flatten().float() for w in weights_s])
    cos_sim = torch.dot(flat_i, flat_s) / (flat_i.norm() * flat_s.norm() + 1e-8)
    return (1.0 - cos_sim.item())


# ═══════════════════════════════════════════════════════════
# HEDONIC GAME (from working implementation logic)
# ═══════════════════════════════════════════════════════════

def compute_hedonic_utility(client_id, coalition, client_model_weights,
                             client_accs, config):
    """
    Hedonic utility using gradient similarity (cosine distance).
    Matches working implementation's utility structure.
    """
    if not coalition:
        return -float("inf")

    λ = config["lambda_perf"]
    β = config["beta_size"]
    γ = config["gamma_grad"]
    μ = config["mu_fair"]

    # Performance
    perf = λ * client_accs.get(client_id, 0.0)

    # Size penalty
    size_pen = β * len(coalition)

    # Gradient similarity using weight distances
    client_w = client_model_weights[client_id]
    sims = []
    for j in coalition:
        if j != client_id:
            dist = compute_gradient_distance(client_w, client_model_weights[j])
            sims.append(1.0 - dist)  # convert distance to similarity
    grad_sim = γ * (np.mean(sims) if sims else 0.0)

    # Fairness
    fairness = μ * min(client_accs.get(j, 0.0) for j in coalition)

    return perf - size_pen + grad_sim + fairness


def form_coalitions_hedonic(num_clients, K, client_model_weights,
                             client_accs, config, prev_coalitions=None):
    """Nash-stable coalition formation."""
    min_size   = config["min_coalition_size"]
    nash_iters = config["nash_iterations"]
    rng        = np.random.RandomState(config.get("round_seed", 42))

    if prev_coalitions is not None:
        c2k = {c: k for k, coal in enumerate(prev_coalitions) for c in coal}
    else:
        shuffled = list(range(num_clients)); rng.shuffle(shuffled)
        c2k = {c: shuffled.index(c) % K for c in range(num_clients)}

    for _ in range(nash_iters):
        improved = False
        order = list(range(num_clients)); rng.shuffle(order)
        for client in order:
            curr_k    = c2k[client]
            curr_coal = [c for c, k in c2k.items() if k == curr_k]
            curr_u    = compute_hedonic_utility(client, curr_coal,
                                                 client_model_weights,
                                                 client_accs, config)
            best_k, best_u = curr_k, curr_u
            for tgt_k in range(K):
                if tgt_k == curr_k: continue
                src_after = [c for c in curr_coal if c != client]
                if 0 < len(src_after) < min_size: continue
                tgt_coal = [c for c, k in c2k.items() if k == tgt_k] + [client]
                u = compute_hedonic_utility(client, tgt_coal,
                                             client_model_weights,
                                             client_accs, config)
                if u > best_u + 1e-6:
                    best_u, best_k = u, tgt_k
            if best_k != curr_k:
                c2k[client] = best_k; improved = True
        if not improved:
            break

    coalitions = [[] for _ in range(K)]
    for c, k in c2k.items():
        coalitions[k].append(c)

    # Enforce min_coalition_size
    final, small = [], []
    for coal in coalitions:
        (final if len(coal) >= min_size else small).extend(
            [coal] if len(coal) >= min_size else coal)
    if small:
        if final: final[0].extend(small)
        else: final = [small]
    while len(final) < K and any(len(c) > min_size * 2 for c in final):
        for i, coal in enumerate(final):
            if len(coal) > min_size * 2:
                mid = len(coal) // 2
                final[i] = coal[:mid]; final.append(coal[mid:]); break
    return final


# ═══════════════════════════════════════════════════════════
# MFG (ported from working fast implementation)
# ═══════════════════════════════════════════════════════════

def compute_mfg_payoff(action, mean_field_action, base_acc,
                        coalition_min_acc, config):
    """
    Fast approximate payoff — no full training needed.
    Directly ported from working implementation.
    """
    epochs, participation = action
    mean_epochs, mean_part = mean_field_action

    alpha_comp  = config["alpha_comp"]
    beta_part   = config["beta_part"]
    gamma_sync  = config["gamma_sync"]
    lambda_fair = config["lambda_fair_mfg"]
    delta_contrib = config["delta_contrib"]

    # Performance: current acc scaled by effort
    effort = (epochs / 10.0) * participation
    est_acc = min(1.0, base_acc * (1.0 + 0.05 * effort))

    # Costs
    comp_cost  = alpha_comp * epochs
    part_cost  = beta_part * (1.0 - participation)
    action_vec = np.array([epochs, participation])
    mean_vec   = np.array([mean_epochs, mean_part])
    sync_pen   = gamma_sync * np.linalg.norm(action_vec - mean_vec) ** 2

    # Bonuses
    fairness    = lambda_fair * coalition_min_acc
    contribution = delta_contrib * (epochs / 20.0)

    return est_acc - comp_cost - part_cost - sync_pen + fairness + contribution


def solve_mfg_fast(coalition, client_accs, config):
    """
    Fast MFG with coarse grid — directly from working implementation.
    epoch_space: 5 options, participation: 3 options = 15 actions total.
    """
    # Coarse action space from working implementation
    epoch_space       = [3, 5, 7, 10, 15]
    participation_space = [0.85, 0.9, 1.0]  # min 0.85 to prevent degradation
    max_iter          = config["mfg_iterations"]

    # Initialize
    client_actions = {c: (config["local_epochs_base"], 1.0) for c in coalition}

    # Coalition min accuracy
    accs = [client_accs.get(c, 0.5) for c in coalition]
    coalition_min_acc = min(accs) if accs else 0.5

    for iteration in range(max_iter):
        mean_epochs = np.mean([a[0] for a in client_actions.values()])
        mean_part   = np.mean([a[1] for a in client_actions.values()])
        mean_field  = (mean_epochs, mean_part)
        old_actions = client_actions.copy()

        for c in coalition:
            base_acc  = client_accs.get(c, 0.5)
            best_pay  = -float('inf')
            best_act  = client_actions[c]

            for e in epoch_space:
                for tau in participation_space:
                    pay = compute_mfg_payoff(
                        (e, tau), mean_field, base_acc,
                        coalition_min_acc, config)
                    if pay > best_pay:
                        best_pay, best_act = pay, (e, tau)

            client_actions[c] = best_act

        # Check convergence
        changes = [abs(client_actions[c][0] - old_actions[c][0]) +
                   abs(client_actions[c][1] - old_actions[c][1])
                   for c in coalition]
        if max(changes) < 0.5:
            break

    return client_actions  # {client_id: (epochs, participation)}


# ═══════════════════════════════════════════════════════════
# LOCAL TRAINING
# ═══════════════════════════════════════════════════════════

def local_train_client(model, dataloader, epochs, lr, momentum,
                        weight_decay, device,
                        personal_head=None, finetune_epochs=3):
    """Standard local SGD training — clean, no participation skipping."""
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                           momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    # Fine-tune personal head
    if personal_head is not None:
        model.eval()
        personal_head = personal_head.to(device)
        personal_head.train()
        head_opt = optim.Adam(personal_head.parameters(), lr=lr * 0.5)
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
    print("\n" + "="*50)
    print("  Running: HedonicMFG (Improved)")
    print("="*50)

    tracker           = ResultsTracker("HedonicMFG")
    num_clients       = config["num_clients"]
    total_rounds      = config["total_rounds"]
    K                 = config["num_coalitions"]
    warmup_rounds     = config["warmup_rounds"]
    recluster_interval= config["recluster_interval"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]
    total_data        = sum(client_data_sizes)
    use_ph            = config.get("use_personalized_head", True)

    # ── Phase 1: FedAvg Warmup ──────────────────────────
    print(f"\n  [Phase 1] Warmup ({warmup_rounds} rounds) ...")
    global_model = get_model(config).to(device)

    # Track individual client models (KEY from working implementation)
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
                device=device)
            client_models[i] = lm
            local_models.append(lm)

        weights    = [s / total_data for s in client_data_sizes]
        agg_params = fedavg_aggregate(local_models, weights)
        for p, ap in zip(global_model.parameters(), agg_params):
            p.data.copy_(ap)

    warmup_acc = evaluate_model(global_model, global_test_loader, device)
    print(f"  [Phase 1] Warmup complete. Global acc: {warmup_acc*100:.2f}%")

    # ── Phase 2: Coalition formation ─────────────────────
    print(f"\n  [Phase 2] Initial coalition formation (K={K})...")

    # Client accuracies
    client_accs_dict = {
        i: evaluate_model(client_models[i], client_val_loaders[i], device)
        for i in range(num_clients)
    }

    # Client model weights for hedonic utility
    client_model_weights = {
        i: [p.data.clone().cpu() for p in client_models[i].parameters()]
        for i in range(num_clients)
    }

    config["round_seed"] = config["seed"]
    coalitions = form_coalitions_hedonic(
        num_clients, K, client_model_weights, client_accs_dict, config)
    print(f"  Coalitions: {coalitions}")

    # Coalition models = aggregate of member client models (from working impl)
    c2k = {c: k for k, coal in enumerate(coalitions) for c in coal}
    coalition_models = {}
    for k, coal in enumerate(coalitions):
        members     = [client_models[c] for c in coal]
        coal_sizes  = [client_data_sizes[c] for c in coal]
        total_coal  = sum(coal_sizes)
        agg = fedavg_aggregate(members, [s/total_coal for s in coal_sizes])
        coalition_models[k] = copy.deepcopy(global_model)
        for p, ap in zip(coalition_models[k].parameters(), agg):
            p.data.copy_(ap)

    personal_heads = {i: get_personal_head(config).to(device)
                      for i in range(num_clients)} if use_ph else {}

    # ── Phase 3: Clustered training ───────────────────────
    print(f"\n  [Phase 3] Clustered training with MFG "
          f"({total_rounds - warmup_rounds} rounds)...")

    for rnd in range(warmup_rounds + 1, total_rounds + 1):
        t0 = time.time()
        config["round_seed"] = config["seed"] + rnd

        # Reclustering
        if (rnd - warmup_rounds) > 1 and \
           (rnd - warmup_rounds - 1) % recluster_interval == 0:

            client_accs_dict = {
                i: evaluate_model(client_models[i], client_val_loaders[i], device)
                for i in range(num_clients)
            }
            client_model_weights = {
                i: [p.data.clone().cpu() for p in client_models[i].parameters()]
                for i in range(num_clients)
            }
            old = coalitions
            coalitions = form_coalitions_hedonic(
                num_clients, K, client_model_weights,
                client_accs_dict, config, prev_coalitions=old)
            new_c2k = {c: k for k, coal in enumerate(coalitions) for c in coal}

            # Rebuild coalition models from current member models
            new_cmodels = {}
            for k, coal in enumerate(coalitions):
                members    = [client_models[c] for c in coal]
                coal_sizes = [client_data_sizes[c] for c in coal]
                total_coal = sum(coal_sizes)
                agg = fedavg_aggregate(members, [s/total_coal for s in coal_sizes])
                new_cmodels[k] = copy.deepcopy(global_model)
                for p, ap in zip(new_cmodels[k].parameters(), agg):
                    p.data.copy_(ap)

            coalition_models = new_cmodels
            c2k = new_c2k

        # Solve MFG per coalition
        all_strats = {}
        for k, coal in enumerate(coalitions):
            coal_accs = {c: client_accs_dict.get(c, 0.5) for c in coal}
            strats = solve_mfg_fast(coal, coal_accs, config)
            all_strats.update(strats)

        # Local training — each client trains from their coalition model
        cu = {k: [] for k in range(len(coalitions))}
        cw = {k: [] for k in range(len(coalitions))}

        for i in range(num_clients):
            k = c2k[i]
            # Start from coalition model (not global — key difference)
            lm = copy.deepcopy(coalition_models[k])
            opt_e, _ = all_strats.get(i, (config["local_epochs_base"], 1.0))
            ph = personal_heads.get(i) if use_ph else None

            trained, trained_head = local_train_client(
                lm, client_train_loaders[i],
                epochs=opt_e,
                lr=config["lr"], momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device,
                personal_head=copy.deepcopy(ph) if ph is not None else None,
                finetune_epochs=config.get("finetune_rounds", 3))

            # Update individual client model
            client_models[i] = trained
            cu[k].append(trained)
            cw[k].append(client_data_sizes[i])
            if trained_head is not None:
                personal_heads[i] = trained_head

        # Coalition aggregation
        for k in range(len(coalitions)):
            if cu[k]:
                tw  = sum(cw[k])
                agg = fedavg_aggregate(cu[k], [w/tw for w in cw[k]])
                for p, ap in zip(coalition_models[k].parameters(), agg):
                    p.data.copy_(ap)

        # Meta-aggregation
        coal_sizes = [sum(client_data_sizes[c] for c in coal) for coal in coalitions]
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
                                 global_acc, sum(c_accs_list)/len(c_accs_list))

    print(f"\n  [HedonicMFG] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
