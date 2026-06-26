"""
algorithms/fmnist/hedonic_mfg_fmnist.py — HedonicMFG for FashionMNIST

Uses the proven working architecture:
  - Individual client model tracking
  - Coalition model = weighted aggregate of member models
  - Weight-space similarity for hedonic utility
  - Fast MFG coarse grid (tau >= 0.85 always)
  - Personal 2-layer head fine-tuned per client
  - K=3 coalitions: clothing/footwear/accessories naturally separate

FashionMNIST specific tuning:
  - High gamma_grad=0.9: CNN weight similarity is very informative
    for fashion categories (conv filters specialise per category)
  - lambda_fair_mfg=0.5: moderate fairness — hard classes (shirt,coat)
    need coalition support
  - recluster_interval=20: fashion gradients stable, less reclustering needed
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import numpy as np

from utils import (ResultsTracker, print_round_summary,
                    evaluate_model, evaluate_with_head, fedavg_aggregate)
from local_trainer_fmnist import local_train_fmnist_hedonic
from models_fmnist import get_fmnist_model, get_fmnist_personal_head


# ═══════════════════════════════════════════════════════════
# HEDONIC GAME — weight-space similarity
# ═══════════════════════════════════════════════════════════

def compute_weight_similarity(weights_i: list, weights_j: list) -> float:
    """Cosine similarity between flattened model weight vectors."""
    flat_i = torch.cat([w.flatten().float() for w in weights_i])
    flat_j = torch.cat([w.flatten().float() for w in weights_j])
    return (torch.dot(flat_i, flat_j) /
            (flat_i.norm() * flat_j.norm() + 1e-8)).item()


def compute_hedonic_utility(client_id, coalition, client_model_weights,
                             client_accs, config):
    if not coalition:
        return -float("inf")

    perf     = config["lambda_perf"] * client_accs.get(client_id, 0.0)
    size_pen = config["beta_size"] * len(coalition)

    client_w = client_model_weights[client_id]
    sims     = [compute_weight_similarity(client_w, client_model_weights[j])
                for j in coalition if j != client_id]
    grad_sim = config["gamma_grad"] * (np.mean(sims) if sims else 0.0)
    fairness = config["mu_fair"] * min(client_accs.get(j, 0.0) for j in coalition)

    return perf - size_pen + grad_sim + fairness


def form_coalitions(num_clients, K, client_model_weights,
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

    # Enforce min coalition size
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
# MEAN FIELD GAME
# ═══════════════════════════════════════════════════════════

def mfg_payoff(action, mean_field, base_acc, coalition_min_acc, config):
    epochs, tau = action
    mean_e, mean_t = mean_field

    effort    = (epochs / 10.0) * tau
    est_acc   = min(1.0, base_acc * (1.0 + 0.05 * effort))
    comp_cost = config["alpha_comp"] * epochs
    part_cost = config["beta_part"] * (1.0 - tau)
    sync_pen  = config["gamma_sync"] * ((epochs - mean_e)**2 + (tau - mean_t)**2)
    fairness  = config["lambda_fair_mfg"] * coalition_min_acc
    contrib   = config["delta_contrib"] * (epochs / 20.0)

    return est_acc - comp_cost - part_cost - sync_pen + fairness + contrib


def solve_mfg(coalition, client_accs, config):
    """Fast MFG with coarse grid. tau >= 0.85 always."""
    epoch_space = [3, 5, 7, 10]
    tau_space   = [0.85, 0.9, 1.0]
    e_base      = config["local_epochs_base"]
    strategies  = {c: (e_base, 1.0) for c in coalition}

    coal_min = min(client_accs.get(c, 0.5) for c in coalition)

    for _ in range(config["mfg_iterations"]):
        mean_e = np.mean([strategies[c][0] for c in coalition])
        mean_t = np.mean([strategies[c][1] for c in coalition])
        mf     = (mean_e, mean_t)
        old    = strategies.copy()

        for c in coalition:
            base = client_accs.get(c, 0.5)
            be, bt = strategies[c]
            bu = mfg_payoff((be, bt), mf, base, coal_min, config)
            for e in epoch_space:
                for tau in tau_space:
                    u = mfg_payoff((e, tau), mf, base, coal_min, config)
                    if u > bu + 1e-6:
                        bu, be, bt = u, e, tau
            strategies[c] = (be, bt)

        if strategies == old:
            break
    return strategies


# ═══════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════

def run_hedonic_mfg_fmnist(config, client_train_loaders, client_val_loaders,
                            global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: HedonicMFG [FashionMNIST]")
    print("="*50)

    tracker      = ResultsTracker("HedonicMFG")
    N            = config["num_clients"]
    total_rounds = config["total_rounds"]
    K            = config["num_coalitions"]
    R            = config["recluster_interval"]
    sizes        = [len(l.dataset) for l in client_train_loaders]
    total_data   = sum(sizes)
    use_ph       = config.get("use_personalized_head", True)
    warmup       = min(config["warmup_rounds"], max(0, total_rounds - 1))

    # ── Phase 1: FedAvg Warmup ──────────────────────────
    print(f"\n  [Phase 1] Warmup ({warmup} rounds) ...")
    global_model  = get_fmnist_model(config).to(device)
    client_models = {i: copy.deepcopy(global_model) for i in range(N)}

    for rnd in range(1, warmup + 1):
        local_models = []
        for i in range(N):
            lm = copy.deepcopy(global_model)
            lm, _ = local_train_fmnist_hedonic(
                lm, client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device)
            client_models[i] = lm
            local_models.append(lm)

        agg = fedavg_aggregate(local_models, [s/total_data for s in sizes])
        for p, ap in zip(global_model.parameters(), agg):
            p.data.copy_(ap)

    warmup_acc = evaluate_model(global_model, global_test_loader, device)
    print(f"  [Phase 1] Warmup complete. Global acc: {warmup_acc*100:.2f}%")

    # ── Phase 2: Coalition formation ─────────────────────
    print(f"\n  [Phase 2] Initial coalition formation (K={K})...")
    client_accs_dict = {
        i: evaluate_model(client_models[i], client_val_loaders[i], device)
        for i in range(N)
    }
    client_model_weights = {
        i: [p.data.clone().cpu() for p in client_models[i].parameters()]
        for i in range(N)
    }

    config["round_seed"] = config["seed"]
    coalitions = form_coalitions(N, K, client_model_weights,
                                  client_accs_dict, config)
    print(f"  Coalitions: {coalitions}")
    print(f"  Sizes: {[len(c) for c in coalitions]}")

    # Coalition model = aggregate of member models
    c2k = {c: k for k, coal in enumerate(coalitions) for c in coal}
    coalition_models = {}
    for k, coal in enumerate(coalitions):
        members    = [client_models[c] for c in coal]
        coal_sizes = [sizes[c] for c in coal]
        total_coal = sum(coal_sizes)
        agg = fedavg_aggregate(members, [s/total_coal for s in coal_sizes])
        coalition_models[k] = copy.deepcopy(global_model)
        for p, ap in zip(coalition_models[k].parameters(), agg):
            p.data.copy_(ap)

    personal_heads = {i: get_fmnist_personal_head(config).to(device)
                      for i in range(N)} if use_ph else {}

    # ── Phase 3: Clustered training ───────────────────────
    print(f"\n  [Phase 3] Clustered training with MFG "
          f"({total_rounds - warmup} rounds)...")

    for rnd in range(warmup + 1, total_rounds + 1):
        t0 = time.time()
        config["round_seed"] = config["seed"] + rnd

        # Reclustering
        if (rnd - warmup) > 1 and (rnd - warmup - 1) % R == 0:
            client_accs_dict = {
                i: evaluate_model(client_models[i], client_val_loaders[i], device)
                for i in range(N)
            }
            client_model_weights = {
                i: [p.data.clone().cpu() for p in client_models[i].parameters()]
                for i in range(N)
            }
            old = coalitions
            coalitions = form_coalitions(N, K, client_model_weights,
                                          client_accs_dict, config,
                                          prev_coalitions=old)
            new_c2k     = {c: k for k, coal in enumerate(coalitions) for c in coal}
            new_cmodels = {}
            for k, coal in enumerate(coalitions):
                members    = [client_models[c] for c in coal]
                coal_sizes = [sizes[c] for c in coal]
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
            all_strats.update(solve_mfg(coal, coal_accs, config))

        # Local training — from coalition model, save to client_models
        cu = {k: [] for k in range(len(coalitions))}
        cw = {k: [] for k in range(len(coalitions))}

        for i in range(N):
            k = c2k[i]
            lm = copy.deepcopy(coalition_models[k])
            opt_e, _ = all_strats.get(i, (config["local_epochs_base"], 1.0))
            ph = personal_heads.get(i) if use_ph else None

            trained, trained_head = local_train_fmnist_hedonic(
                lm, client_train_loaders[i],
                epochs=opt_e,
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device,
                personal_head=copy.deepcopy(ph) if ph is not None else None,
                finetune_epochs=config.get("finetune_rounds", 3))

            client_models[i] = trained
            cu[k].append(trained)
            cw[k].append(sizes[i])
            if trained_head is not None:
                personal_heads[i] = trained_head

        # Coalition aggregation
        for k in range(len(coalitions)):
            if cu[k]:
                tw  = sum(cw[k])
                agg = fedavg_aggregate(cu[k], [w/tw for w in cw[k]])
                for p, ap in zip(coalition_models[k].parameters(), agg):
                    p.data.copy_(ap)

        # Meta-aggregation → global model
        coal_sizes = [sum(sizes[c] for c in coal) for coal in coalitions]
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
        for i in range(N):
            k = c2k[i]
            if use_ph and i in personal_heads:
                acc = evaluate_with_head(coalition_models[k], personal_heads[i],
                                          client_val_loaders[i], device)
            else:
                acc = evaluate_model(client_models[i],
                                      client_val_loaders[i], device)
            c_accs_list.append(acc)

        client_accs_dict = {i: c_accs_list[i] for i in range(N)}
        tracker.log(rnd, global_acc, c_accs_list, time.time() - t0)

        if rnd % 10 == 0 or rnd == warmup + 1:
            print_round_summary("HedonicMFG", rnd, total_rounds,
                                 global_acc, sum(c_accs_list)/N)

    if tracker.global_accs:
        print(f"\n  [HedonicMFG] Best global: {max(tracker.global_accs)*100:.2f}%  "
              f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
