"""
algorithms/hedonic_mfg.py — HedonicMFG (Full Improved Version)

Improvements over paper baseline:
  1. K=3 coalitions for N=20 clients (stable ~6-7 clients/coalition)
  2. Long warmup (T_w=15) so gradients are informative before coalition formation
  3. Adaptive reclustering triggered by cosine similarity drop (not just fixed R)
  4. Per-client personalized classification heads (fine-tuned on local data)
  5. Tuned MFG utility weights: high λ_fair for MNIST fairness, tuned γ_sync
  6. Min coalition size constraint prevents singleton coalitions
  7. MFG epoch range [3, 10] per-client individualized training
"""

import torch
import torch.nn as nn
import copy
import time
import numpy as np
from utils import (ResultsTracker, print_round_summary, evaluate_model,
                    evaluate_with_head, fedavg_aggregate,
                    pairwise_cosine_similarity, get_model_params,
                    cosine_similarity)
from local_trainer import local_train_hedonic
from models import get_model, get_personal_head


# ═══════════════════════════════════════════════════════════
# HEDONIC GAME — Coalition Formation
# ═══════════════════════════════════════════════════════════

def compute_hedonic_utility(client_id: int, coalition: list,
                             client_accs: dict, grad_sim_matrix: np.ndarray,
                             config: dict) -> float:
    """
    Compute hedonic utility ϕ_i(S_k) for client i in coalition S.

    ϕ_i(S) = λ_perf * Acc_i
            - β_size * |S|
            + γ_grad * avg_cosine_similarity(i, S)
            + μ_fair * min_j Acc_j(in S)
    """
    if len(coalition) == 0:
        return -float("inf")

    λ_perf  = config["lambda_perf"]
    β_size  = config["beta_size"]
    γ_grad  = config["gamma_grad"]
    μ_fair  = config["mu_fair"]

    # Performance gain
    perf = λ_perf * client_accs.get(client_id, 0.0)

    # Size penalty
    size_pen = β_size * len(coalition)

    # Gradient similarity: avg cosine sim between client_id and coalition members
    sims = [grad_sim_matrix[client_id][j]
            for j in coalition if j != client_id]
    grad_sim = γ_grad * (np.mean(sims) if sims else 0.0)

    # Fairness: encourage helping weakest member
    fairness = μ_fair * min(client_accs.get(j, 0.0) for j in coalition)

    return perf - size_pen + grad_sim + fairness


def form_coalitions_hedonic(num_clients: int, K: int,
                             client_accs: dict,
                             grad_sim_matrix: np.ndarray,
                             config: dict,
                             prev_coalitions: list = None) -> list:
    """
    Hedonic game coalition formation.

    Algorithm:
    1. Initialize with prev_coalitions or random assignment
    2. For each client, check if switching to another coalition improves utility
    3. Repeat until Nash-stable (no beneficial switch) or max iterations
    4. Enforce min_coalition_size constraint

    Returns: list of lists (coalition → client_ids)
    """
    min_size = config["min_coalition_size"]
    nash_iters = config["nash_iterations"]
    seed = config.get("round_seed", 42)
    rng = np.random.RandomState(seed)

    # Initialization
    if prev_coalitions is not None:
        # Warm-start from previous coalitions
        client_to_coalition = {}
        for k, coal in enumerate(prev_coalitions):
            for c in coal:
                client_to_coalition[c] = k
    else:
        # Random initialization
        shuffled = list(range(num_clients))
        rng.shuffle(shuffled)
        client_to_coalition = {c: shuffled.index(c) % K for c in range(num_clients)}

    def get_coalitions():
        result = [[] for _ in range(K)]
        for c, k in client_to_coalition.items():
            result[k].append(c)
        return result

    # Nash stability iterations
    for iteration in range(nash_iters):
        improved = False
        client_order = list(range(num_clients))
        rng.shuffle(client_order)

        for client in client_order:
            current_k = client_to_coalition[client]
            current_coal = [c for c, k in client_to_coalition.items() if k == current_k]
            current_utility = compute_hedonic_utility(
                client, current_coal, client_accs, grad_sim_matrix, config
            )

            best_k = current_k
            best_utility = current_utility

            for target_k in range(K):
                if target_k == current_k:
                    continue
                target_coal = [c for c, k in client_to_coalition.items()
                               if k == target_k]

                # Check min size constraint on source coalition
                source_after = [c for c in current_coal if c != client]
                if len(source_after) < min_size and len(source_after) > 0:
                    continue  # Don't leave source too small

                candidate_coal = target_coal + [client]
                utility = compute_hedonic_utility(
                    client, candidate_coal, client_accs, grad_sim_matrix, config
                )

                if utility > best_utility + 1e-6:
                    best_utility = utility
                    best_k = target_k

            if best_k != current_k:
                client_to_coalition[client] = best_k
                improved = True

        if not improved:
            break  # Nash stable

    coalitions = get_coalitions()

    # Enforce min_coalition_size: merge too-small coalitions
    final_coalitions = []
    small = []
    for coal in coalitions:
        if len(coal) >= min_size:
            final_coalitions.append(coal)
        else:
            small.extend(coal)

    if small:
        if final_coalitions:
            final_coalitions[0].extend(small)
        else:
            final_coalitions = [small]

    # Ensure K coalitions (split large ones if needed)
    while len(final_coalitions) < K and any(len(c) > min_size * 2
                                             for c in final_coalitions):
        for i, coal in enumerate(final_coalitions):
            if len(coal) > min_size * 2:
                mid = len(coal) // 2
                final_coalitions[i] = coal[:mid]
                final_coalitions.append(coal[mid:])
                break

    return final_coalitions


# ═══════════════════════════════════════════════════════════
# MEAN FIELD GAME — Training Strategy Optimization
# ═══════════════════════════════════════════════════════════

def approximate_acc_improvement(current_acc: float, epochs: int,
                                  participation: float, e_base: int,
                                  tau_base: float = 1.0, eta: float = 0.15) -> float:
    """
    Linear approximation: Acc_i(E, τ) ≈ Acc_i(current) * (1 + η * E*τ / (E_base*τ_base))
    Capped at 0.99 to avoid unrealistic estimates.
    """
    improvement = current_acc * (1.0 + eta * (epochs * participation)
                                  / (e_base * tau_base + 1e-8))
    return min(improvement, 0.99)


def mfg_utility(client_id: int, epochs: int, participation: float,
                 mean_E: float, mean_tau: float,
                 current_acc: float, coalition: list,
                 client_accs: dict, config: dict) -> float:
    """
    MFG utility for client i:
    u_i = Acc_i(E,τ) - α_comp*E - β_part*(1-τ)
          - γ_sync*((E-Ē)² + (τ-τ̄)²)
          + λ_fair*min_j Acc_j
          + δ_contrib*(E*τ/|S|)
    """
    α = config["alpha_comp"]
    β = config["beta_part"]
    γ = config["gamma_sync"]
    λ = config["lambda_fair_mfg"]
    δ = config["delta_contrib"]
    e_base = config["local_epochs_base"]

    acc = approximate_acc_improvement(current_acc, epochs, participation,
                                       e_base)
    comp_cost = α * epochs
    part_penalty = β * (1.0 - participation)
    sync_cost = γ * ((epochs - mean_E) ** 2 + (participation - mean_tau) ** 2)
    fairness = λ * min(client_accs.get(j, 0.0) for j in coalition)
    contrib = δ * (epochs * participation) / (len(coalition) + 1e-8)

    return acc - comp_cost - part_penalty - sync_cost + fairness + contrib


def solve_mfg(coalition: list, client_accs: dict,
               config: dict) -> dict:
    """
    Solve Mean Field Game within a coalition.
    Returns dict: client_id → (optimal_epochs, optimal_participation)

    Algorithm:
    1. Initialize with base strategy
    2. Compute mean field (Ē, τ̄)
    3. Each client best-responds to mean field
    4. Repeat until convergence (MFG equilibrium)
    """
    e_min = config["e_min"]
    e_max = config["e_max"]
    e_base = config["local_epochs_base"]
    mfg_iters = config["mfg_iterations"]

    # Candidate action spaces
    epoch_candidates = list(range(e_min, e_max + 1, 1))
    tau_candidates = [0.5, 0.7, 0.85, 1.0]

    # Initialize strategies
    strategies = {c: (e_base, 1.0) for c in coalition}

    for _ in range(mfg_iters):
        # Compute mean field
        all_epochs = [strategies[c][0] for c in coalition]
        all_taus = [strategies[c][1] for c in coalition]
        mean_E = np.mean(all_epochs)
        mean_tau = np.mean(all_taus)

        # Each client best-responds
        new_strategies = {}
        for c in coalition:
            best_e, best_tau = strategies[c]
            best_u = mfg_utility(c, best_e, best_tau, mean_E, mean_tau,
                                  client_accs.get(c, 0.5), coalition,
                                  client_accs, config)

            for e in epoch_candidates:
                for tau in tau_candidates:
                    u = mfg_utility(c, e, tau, mean_E, mean_tau,
                                     client_accs.get(c, 0.5), coalition,
                                     client_accs, config)
                    if u > best_u + 1e-6:
                        best_u = u
                        best_e, best_tau = e, tau

            new_strategies[c] = (best_e, best_tau)

        # Check convergence
        if new_strategies == strategies:
            break
        strategies = new_strategies

    return strategies  # {client_id: (epochs, participation_rate)}


# ═══════════════════════════════════════════════════════════
# HEDONICMFG MAIN RUNNER
# ═══════════════════════════════════════════════════════════

def run_hedonic_mfg(config, client_train_loaders, client_val_loaders,
                    global_test_loader, device):
    print("\n" + "="*50)
    print("  Running: HedonicMFG (Improved)")
    print("="*50)

    tracker = ResultsTracker("HedonicMFG")
    num_clients = config["num_clients"]
    total_rounds = config["total_rounds"]
    K = config["num_coalitions"]
    warmup_rounds = config["warmup_rounds"]
    recluster_interval = config["recluster_interval"]
    client_data_sizes = [len(l.dataset) for l in client_train_loaders]
    total_data = sum(client_data_sizes)
    use_personal_head = config.get("use_personalized_head", True)

    # ── Phase 1: Warmup with FedAvg ──────────────────────
    print(f"\n  [Phase 1] Warmup ({warmup_rounds} rounds) ...")
    global_model = get_model(config).to(device)

    for rnd in range(1, warmup_rounds + 1):
        local_models = []
        for i in range(num_clients):
            lm = copy.deepcopy(global_model)
            lm = local_train_hedonic(
                lm, client_train_loaders[i],
                epochs=config["local_epochs_base"],
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device
            )[0]
            local_models.append(lm)

        weights = [client_data_sizes[i] / total_data for i in range(num_clients)]
        agg_params = fedavg_aggregate(local_models, weights)
        for p, ap in zip(global_model.parameters(), agg_params):
            p.data.copy_(ap)

    print(f"  [Phase 1] Warmup complete. Global acc: "
          f"{evaluate_model(global_model, global_test_loader, device)*100:.2f}%")

    # ── Compute initial gradient similarity ───────────────
    def compute_grad_sim_matrix():
        """Run one forward+backward pass per client, collect gradient vectors."""
        grad_vecs = []
        criterion = nn.CrossEntropyLoss()
        for i in range(num_clients):
            tmp = copy.deepcopy(global_model).to(device)
            tmp.train()
            for x, y in client_train_loaders[i]:
                x, y = x.to(device), y.to(device)
                tmp.zero_grad()
                out = tmp(x)
                loss = criterion(out, y)
                loss.backward()
                break  # One batch is enough
            grads = torch.cat([
                p.grad.data.clone().flatten()
                for p in tmp.parameters()
                if p.grad is not None
            ])
            grad_vecs.append(grads)
        return pairwise_cosine_similarity(grad_vecs)

    # ── Phase 2: Compute initial client accuracies and form coalitions ──
    print(f"\n  [Phase 2] Initial coalition formation (K={K})...")
    client_accs_dict = {
        i: evaluate_model(global_model, client_val_loaders[i], device)
        for i in range(num_clients)
    }
    grad_sim_matrix = compute_grad_sim_matrix()
    coalitions = form_coalitions_hedonic(
        num_clients, K, client_accs_dict, grad_sim_matrix, config
    )
    print(f"  Coalitions: {coalitions}")

    # Initialize coalition models and personal heads
    coalition_models = {k: copy.deepcopy(global_model) for k in range(len(coalitions))}
    personal_heads = {
        i: get_personal_head(config).to(device)
        for i in range(num_clients)
    } if use_personal_head else {}

    client_to_coalition = {}
    for k, coal in enumerate(coalitions):
        for c in coal:
            client_to_coalition[c] = k

    # Adaptive reclustering: track within-coalition similarity
    last_sim_score = 1.0

    # ── Phase 3: Clustered training with MFG ──────────────
    print(f"\n  [Phase 3] Clustered training with MFG ({total_rounds - warmup_rounds} rounds)...")

    for rnd in range(warmup_rounds + 1, total_rounds + 1):
        t0 = time.time()
        config["round_seed"] = config["seed"] + rnd

        # ── Adaptive reclustering ──────────────────────────
        should_recluster = ((rnd - warmup_rounds) % recluster_interval == 0)

        if should_recluster:
            # Recompute similarity to check if reclustering is needed
            grad_sim_matrix = compute_grad_sim_matrix()
            client_accs_dict = {
                i: evaluate_model(coalition_models[client_to_coalition[i]],
                                   client_val_loaders[i], device)
                for i in range(num_clients)
            }

            # Check within-coalition similarity health
            within_sims = []
            for k, coal in enumerate(coalitions):
                for ci in coal:
                    for cj in coal:
                        if ci < cj:
                            within_sims.append(grad_sim_matrix[ci][cj])
            avg_within_sim = np.mean(within_sims) if within_sims else 1.0

            # Recluster if similarity dropped significantly
            if avg_within_sim < last_sim_score - 0.05 or avg_within_sim < 0.3:
                old_coalitions = coalitions
                coalitions = form_coalitions_hedonic(
                    num_clients, K, client_accs_dict,
                    grad_sim_matrix, config,
                    prev_coalitions=old_coalitions
                )
                # Reassign clients
                new_client_to_coalition = {}
                for k, coal in enumerate(coalitions):
                    for c in coal:
                        new_client_to_coalition[c] = k

                # Inherit coalition models for clients that stayed in same cluster
                new_coalition_models = {}
                for k, coal in enumerate(coalitions):
                    # Use the most common previous coalition model
                    prev_ks = [client_to_coalition.get(c, 0) for c in coal]
                    dominant_prev_k = max(set(prev_ks), key=prev_ks.count)
                    src_k = dominant_prev_k if dominant_prev_k in coalition_models else 0
                    new_coalition_models[k] = copy.deepcopy(coalition_models[src_k])

                coalition_models = new_coalition_models
                client_to_coalition = new_client_to_coalition
                last_sim_score = avg_within_sim

        # ── Solve MFG within each coalition ───────────────
        all_strategies = {}
        for k, coal in enumerate(coalitions):
            coal_accs = {c: client_accs_dict.get(c, 0.5) for c in coal}
            strategies = solve_mfg(coal, coal_accs, config)
            all_strategies.update(strategies)

        # ── Local training with MFG strategies ────────────
        cluster_updates = {k: [] for k in range(len(coalitions))}
        cluster_weights = {k: [] for k in range(len(coalitions))}

        for i in range(num_clients):
            k = client_to_coalition[i]
            local_model = copy.deepcopy(coalition_models[k])
            opt_epochs, opt_tau = all_strategies.get(i, (config["local_epochs_base"], 1.0))
            ph = personal_heads.get(i) if use_personal_head else None

            trained_model, trained_head = local_train_hedonic(
                local_model,
                client_train_loaders[i],
                epochs=opt_epochs,
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                device=device,
                participation_rate=opt_tau,
                personal_head=copy.deepcopy(ph) if ph is not None else None,
                finetune_epochs=config.get("finetune_rounds", 2)
            )

            cluster_updates[k].append(trained_model)
            cluster_weights[k].append(client_data_sizes[i])
            if trained_head is not None:
                personal_heads[i] = trained_head

        # ── Coalition-level aggregation ────────────────────
        for k in range(len(coalitions)):
            if cluster_updates[k]:
                tw = sum(cluster_weights[k])
                nw = [w / tw for w in cluster_weights[k]]
                agg_params = fedavg_aggregate(cluster_updates[k], nw)
                for p, ap in zip(coalition_models[k].parameters(), agg_params):
                    p.data.copy_(ap)

        # ── Meta-aggregation → global model ───────────────
        coal_sizes = [
            sum(client_data_sizes[c] for c in coal)
            for coal in coalitions
        ]
        total_coal = sum(coal_sizes)
        valid_ks = [k for k, s in enumerate(coal_sizes) if s > 0 and k in coalition_models]

        global_model_eval = copy.deepcopy(coalition_models[valid_ks[0]])
        if len(valid_ks) > 1:
            agg_params = fedavg_aggregate(
                [coalition_models[k] for k in valid_ks],
                [coal_sizes[k] / total_coal for k in valid_ks]
            )
            for p, ap in zip(global_model_eval.parameters(), agg_params):
                p.data.copy_(ap)

        # ── Evaluate ───────────────────────────────────────
        global_acc = evaluate_model(global_model_eval, global_test_loader, device)

        # Client accuracy: use personal head if available, else coalition model
        client_accs_list = []
        for i in range(num_clients):
            k = client_to_coalition[i]
            if use_personal_head and i in personal_heads:
                acc = evaluate_with_head(
                    coalition_models[k], personal_heads[i],
                    client_val_loaders[i], device
                )
            else:
                acc = evaluate_model(coalition_models[k],
                                      client_val_loaders[i], device)
            client_accs_list.append(acc)

        # Update client accs dict for next round's MFG
        client_accs_dict = {i: client_accs_list[i] for i in range(num_clients)}

        elapsed = time.time() - t0
        tracker.log(rnd, global_acc, client_accs_list, elapsed)

        if rnd % 10 == 0 or rnd == warmup_rounds + 1:
            print_round_summary("HedonicMFG", rnd, total_rounds,
                                 global_acc, sum(client_accs_list) / len(client_accs_list))

    print(f"\n  [HedonicMFG] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
