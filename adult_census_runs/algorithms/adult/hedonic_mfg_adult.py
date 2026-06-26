"""
algorithms/adult/hedonic_mfg_adult.py — HedonicMFG for Adult Census

Adult-specific improvements for maximum advantage:
  1. K=4 coalitions for N=25 clients (~6/coalition) — stable tabular clusters
  2. T_w=20 warmup — tabular MLP gradients need more rounds to be informative
  3. High λ_fair_mfg=0.8 — closes the global-client accuracy gap on skewed income data
  4. High γ_sync=0.3 — tabular clients with different feature scales need coordination
  5. Two-layer personal head (not linear) — captures non-linear demographic patterns
  6. Per-client weighted CE loss in backbone — handles class imbalance per client
  7. Adaptive reclustering on similarity drop — income clusters drift as training evolves
  8. δ_contrib=0.4 — strongly reward data-rich clients (sizes vary 3-5x on Adult)
"""

import torch
import torch.nn as nn
import copy
import time
import numpy as np

from utils import (ResultsTracker, print_round_summary,
                    evaluate_model, evaluate_with_head,
                    fedavg_aggregate, pairwise_cosine_similarity,
                    get_model_params)
from local_trainer_adult import local_train_adult_hedonic
from models_adult import get_adult_model, get_adult_personal_head


# ═══════════════════════════════════════════════════════════
# HEDONIC GAME — Adult-specific utility
# ═══════════════════════════════════════════════════════════

def compute_hedonic_utility_adult(client_id, coalition, client_accs,
                                   grad_sim_matrix, config):
    """
    Hedonic utility for Adult Census.
    Higher gamma_grad (0.9) — tabular feature similarity is the primary
    signal for coalition quality. Clients with similar feature distributions
    (e.g. similar demographic mix) should cluster together.
    """
    if not coalition:
        return -float("inf")

    λ  = config["lambda_perf"]
    β  = config["beta_size"]
    γ  = config["gamma_grad"]
    μ  = config["mu_fair"]

    perf     = λ * client_accs.get(client_id, 0.0)
    size_pen = β * len(coalition)
    sims     = [grad_sim_matrix[client_id][j] for j in coalition if j != client_id]
    grad_sim = γ * (np.mean(sims) if sims else 0.0)
    fairness = μ * min(client_accs.get(j, 0.0) for j in coalition)

    return perf - size_pen + grad_sim + fairness


def form_coalitions_adult(num_clients, K, client_accs,
                           grad_sim_matrix, config,
                           prev_coalitions=None):
    """
    Nash-stable hedonic coalition formation for Adult Census.
    Same algorithm as MNIST version but with Adult-specific utility.
    """
    min_size   = config["min_coalition_size"]
    nash_iters = config["nash_iterations"]
    rng        = np.random.RandomState(config.get("round_seed", 42))

    if prev_coalitions is not None:
        c2k = {}
        for k, coal in enumerate(prev_coalitions):
            for c in coal: c2k[c] = k
    else:
        shuffled = list(range(num_clients)); rng.shuffle(shuffled)
        c2k = {c: shuffled.index(c) % K for c in range(num_clients)}

    def get_coalitions():
        result = [[] for _ in range(K)]
        for c, k in c2k.items(): result[k].append(c)
        return result

    for _ in range(nash_iters):
        improved = False
        order = list(range(num_clients)); rng.shuffle(order)
        for client in order:
            curr_k    = c2k[client]
            curr_coal = [c for c, k in c2k.items() if k == curr_k]
            curr_u    = compute_hedonic_utility_adult(
                client, curr_coal, client_accs, grad_sim_matrix, config)
            best_k, best_u = curr_k, curr_u

            for tgt_k in range(K):
                if tgt_k == curr_k: continue
                src_after = [c for c in curr_coal if c != client]
                if 0 < len(src_after) < min_size: continue
                tgt_coal = [c for c, k in c2k.items() if k == tgt_k] + [client]
                u = compute_hedonic_utility_adult(
                    client, tgt_coal, client_accs, grad_sim_matrix, config)
                if u > best_u + 1e-6:
                    best_u, best_k = u, tgt_k

            if best_k != curr_k:
                c2k[client] = best_k; improved = True
        if not improved: break

    coalitions = get_coalitions()
    # Enforce min_size
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
# MFG — Adult-specific utility
# ═══════════════════════════════════════════════════════════

def mfg_utility_adult(client_id, epochs, tau, mean_E, mean_tau,
                       current_acc, coalition, client_accs, config):
    """
    MFG utility for Adult Census.
    Key differences from MNIST:
      - lambda_fair_mfg=0.8 (high) — income label skew needs strong fairness
      - gamma_sync=0.3 (high) — feature-scaled tabular clients must coordinate
      - delta_contrib=0.4 — data-rich clients rewarded more
    """
    α  = config["alpha_comp"]
    β  = config["beta_part"]
    γ  = config["gamma_sync"]
    λf = config["lambda_fair_mfg"]
    δ  = config["delta_contrib"]
    eb = config["local_epochs_base"]
    η  = 0.12  # slightly lower improvement rate for tabular

    acc = current_acc * (1.0 + η * epochs * tau / (eb + 1e-8))
    acc = min(acc, 0.92)  # tabular ceiling slightly lower

    return (acc
            - α * epochs
            - β * (1.0 - tau)
            - γ * ((epochs - mean_E)**2 + (tau - mean_tau)**2)
            + λf * min(client_accs.get(j, 0.0) for j in coalition)
            + δ * (epochs * tau) / (len(coalition) + 1e-8))


def solve_mfg_adult(coalition, client_accs, config):
    """Solve MFG within a coalition for Adult Census."""
    e_min, e_max = config["e_min"], config["e_max"]
    e_base       = config["local_epochs_base"]
    mfg_iters    = config["mfg_iterations"]

    epoch_cands = list(range(e_min, e_max + 1, 2))  # step 2 for speed
    tau_cands   = [0.6, 0.75, 0.9, 1.0]
    strategies  = {c: (e_base, 1.0) for c in coalition}

    for _ in range(mfg_iters):
        mean_E   = np.mean([strategies[c][0] for c in coalition])
        mean_tau = np.mean([strategies[c][1] for c in coalition])
        new_strat = {}
        for c in coalition:
            be, bt = strategies[c]
            bu = mfg_utility_adult(c, be, bt, mean_E, mean_tau,
                                    client_accs.get(c, 0.5),
                                    coalition, client_accs, config)
            for e in epoch_cands:
                for tau in tau_cands:
                    u = mfg_utility_adult(c, e, tau, mean_E, mean_tau,
                                           client_accs.get(c, 0.5),
                                           coalition, client_accs, config)
                    if u > bu + 1e-6:
                        bu, be, bt = u, e, tau
            new_strat[c] = (be, bt)
        if new_strat == strategies: break
        strategies = new_strat

    return strategies


# ═══════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════

def run_hedonic_mfg_adult(config, client_train_loaders, client_val_loaders,
                           global_test_loader, device, feature_dim):
    print("\n" + "="*50)
    print("  Running: HedonicMFG [Adult Census — Improved]")
    print("="*50)

    tracker      = ResultsTracker("HedonicMFG")
    N            = config["num_clients"]
    total_rounds = config["total_rounds"]
    K            = config["num_coalitions"]
    # FIX: cap warmup so phase 3 always gets at least 1 round
    warmup = min(config["warmup_rounds"], max(0, total_rounds - 1))
    R            = config["recluster_interval"]
    sizes        = [len(l.dataset) for l in client_train_loaders]
    total_data   = sum(sizes)
    use_ph       = config.get("use_personalized_head", True)

    # ── Phase 1: FedAvg Warmup ───────────────────────────
    print(f"\n  [Phase 1] Warmup ({warmup} rounds) ...")
    global_model = get_adult_model(config, feature_dim).to(device)

    for rnd in range(1, warmup + 1):
        lms = []
        for i in range(N):
            lm = copy.deepcopy(global_model)
            lm, _ = local_train_adult_hedonic(
                lm, client_train_loaders[i],
                config["local_epochs_base"],
                config["lr"], config["weight_decay"], device)
            lms.append(lm)
        agg = fedavg_aggregate(lms, [s / total_data for s in sizes])
        for p, ap in zip(global_model.parameters(), agg): p.data.copy_(ap)

    warmup_acc = evaluate_model(global_model, global_test_loader, device)
    print(f"  [Phase 1] Done. Global acc after warmup: {warmup_acc*100:.2f}%")

    # ── Gradient similarity helper ───────────────────────
    def compute_grad_sim():
        criterion = nn.CrossEntropyLoss()
        grad_vecs = []
        for i in range(N):
            tmp = copy.deepcopy(global_model).to(device)
            tmp.train()
            for x, y in client_train_loaders[i]:
                x, y = x.to(device), y.to(device)
                tmp.zero_grad()
                loss = criterion(tmp(x), y)
                loss.backward()
                break
            grads = torch.cat([p.grad.data.clone().flatten()
                                for p in tmp.parameters() if p.grad is not None])
            grad_vecs.append(grads)
        return pairwise_cosine_similarity(grad_vecs)

    # ── Phase 2: Initial coalition formation ─────────────
    print(f"\n  [Phase 2] Coalition formation (K={K}, N={N})...")
    client_accs_dict = {
        i: evaluate_model(global_model, client_val_loaders[i], device)
        for i in range(N)
    }
    grad_sim = compute_grad_sim()
    coalitions = form_coalitions_adult(N, K, client_accs_dict, grad_sim, config)
    print(f"  Coalitions formed: {coalitions}")
    print(f"  Sizes: {[len(c) for c in coalitions]}")

    coalition_models = {k: copy.deepcopy(global_model) for k in range(len(coalitions))}
    personal_heads   = {
        i: get_adult_personal_head(config).to(device)
        for i in range(N)
    } if use_ph else {}

    c2k = {}
    for k, coal in enumerate(coalitions):
        for c in coal: c2k[c] = k

    last_sim = 1.0

    # ── Phase 3: Clustered training with MFG ─────────────
    print(f"\n  [Phase 3] Clustered training ({total_rounds - warmup} rounds)...")

    for rnd in range(warmup + 1, total_rounds + 1):
        t0 = time.time()
        config["round_seed"] = config["seed"] + rnd

        # Adaptive reclustering
        if (rnd - warmup) % R == 0:
            grad_sim = compute_grad_sim()
            client_accs_dict = {
                i: evaluate_model(coalition_models[c2k[i]],
                                   client_val_loaders[i], device)
                for i in range(N)
            }
            within_sims = [
                grad_sim[ci][cj]
                for coal in coalitions
                for ci in coal for cj in coal if ci < cj
            ]
            avg_sim = np.mean(within_sims) if within_sims else 1.0

            if avg_sim < last_sim - 0.05 or avg_sim < 0.25:
                old_coalitions = coalitions
                coalitions = form_coalitions_adult(
                    N, K, client_accs_dict, grad_sim, config,
                    prev_coalitions=old_coalitions)
                new_c2k = {}
                for k, coal in enumerate(coalitions):
                    for c in coal: new_c2k[c] = k

                new_cmodels = {}
                for k, coal in enumerate(coalitions):
                    prev_ks = [c2k.get(c, 0) for c in coal]
                    dom = max(set(prev_ks), key=prev_ks.count)
                    src = dom if dom in coalition_models else 0
                    new_cmodels[k] = copy.deepcopy(coalition_models[src])

                coalition_models = new_cmodels
                c2k = new_c2k
                last_sim = avg_sim

        # Solve MFG per coalition
        all_strats = {}
        for k, coal in enumerate(coalitions):
            coal_accs = {c: client_accs_dict.get(c, 0.5) for c in coal}
            strats = solve_mfg_adult(coal, coal_accs, config)
            all_strats.update(strats)

        # Local training
        cu = {k: [] for k in range(len(coalitions))}
        cw = {k: [] for k in range(len(coalitions))}

        for i in range(N):
            k = c2k[i]
            lm = copy.deepcopy(coalition_models[k])
            e, tau = all_strats.get(i, (config["local_epochs_base"], 1.0))
            ph = personal_heads.get(i) if use_ph else None

            trained, trained_head = local_train_adult_hedonic(
                lm, client_train_loaders[i],
                epochs=e, lr=config["lr"],
                weight_decay=config["weight_decay"],
                device=device,
                participation_rate=tau,
                personal_head=copy.deepcopy(ph) if ph is not None else None,
                finetune_epochs=config.get("finetune_rounds", 5)
            )
            cu[k].append(trained); cw[k].append(sizes[i])
            if trained_head is not None:
                personal_heads[i] = trained_head

        # Coalition aggregation
        for k in range(len(coalitions)):
            if cu[k]:
                tw = sum(cw[k])
                agg = fedavg_aggregate(cu[k], [w/tw for w in cw[k]])
                for p, ap in zip(coalition_models[k].parameters(), agg): p.data.copy_(ap)

        # Meta-aggregation
        coal_sizes = [sum(sizes[c] for c in coal) for coal in coalitions]
        total_coal = sum(coal_sizes)
        valid_ks = [k for k, s in enumerate(coal_sizes) if s > 0 and k in coalition_models]

        global_eval = copy.deepcopy(coalition_models[valid_ks[0]])
        if len(valid_ks) > 1:
            agg = fedavg_aggregate(
                [coalition_models[k] for k in valid_ks],
                [coal_sizes[k] / total_coal for k in valid_ks])
            for p, ap in zip(global_eval.parameters(), agg): p.data.copy_(ap)

        # Evaluate
        g_acc = evaluate_model(global_eval, global_test_loader, device)
        c_accs_list = []
        for i in range(N):
            k = c2k[i]
            if use_ph and i in personal_heads:
                acc = evaluate_with_head(
                    coalition_models[k], personal_heads[i],
                    client_val_loaders[i], device)
            else:
                acc = evaluate_model(coalition_models[k],
                                      client_val_loaders[i], device)
            c_accs_list.append(acc)

        client_accs_dict = {i: c_accs_list[i] for i in range(N)}
        tracker.log(rnd, g_acc, c_accs_list, time.time() - t0)

        if rnd % 10 == 0 or rnd == warmup + 1:
            print_round_summary("HedonicMFG", rnd, total_rounds,
                                 g_acc, sum(c_accs_list)/N)

    print(f"\n  [HedonicMFG] Best global: {max(tracker.global_accs)*100:.2f}%  "
          f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
