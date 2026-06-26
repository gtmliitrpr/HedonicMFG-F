"""
Random Clustering — Adult Census.
Fix: warmup capped at min(warmup, total_rounds - 1) so phase 3
always gets at least 1 round regardless of config values.
"""
import copy, time
import numpy as np
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from local_trainer_adult import local_train_adult_standard
from models_adult import get_adult_model


def random_partition(N, K, seed):
    rng = np.random.RandomState(seed)
    clients = list(range(N))
    rng.shuffle(clients)
    clusters = [[] for _ in range(K)]
    for idx, c in enumerate(clients):
        clusters[idx % K].append(c)
    return [c for c in clusters if c]


def run_random_clustering_adult(config, client_train_loaders, client_val_loaders,
                                 global_test_loader, device, feature_dim):
    print("\n" + "="*50 + "\n  Running: RandomCluster [Adult]\n" + "="*50)
    tracker      = ResultsTracker("RandomCluster")
    N            = config["num_clients"]
    K            = config["random_clustering_K"]
    R            = config["recluster_interval"]
    total_rounds = config["total_rounds"]
    sizes        = [len(l.dataset) for l in client_train_loaders]
    total        = sum(sizes)

    # FIX: cap warmup so phase 3 always gets at least 1 round
    warmup = min(config["warmup_rounds"], max(0, total_rounds - 1))
    if warmup != config["warmup_rounds"]:
        print(f"  [RandomCluster] Warmup capped: {config['warmup_rounds']} → {warmup} "
              f"(total_rounds={total_rounds})")

    global_model = get_adult_model(config, feature_dim).to(device)

    # Phase 1: Warmup with FedAvg
    print(f"  [RandomCluster] Warmup phase: {warmup} rounds")
    for rnd in range(1, warmup + 1):
        lms = []
        for i in range(N):
            lm = copy.deepcopy(global_model)
            lm = local_train_adult_standard(lm, client_train_loaders[i],
                config["local_epochs_base"], config["lr"],
                config["weight_decay"], device)
            lms.append(lm)
        agg = fedavg_aggregate(lms, [s / total for s in sizes])
        for p, ap in zip(global_model.parameters(), agg):
            p.data.copy_(ap)

    # Phase 2: Clustered training
    clusters      = random_partition(N, K, config["seed"])
    cluster_models = {k: copy.deepcopy(global_model) for k in range(len(clusters))}
    c2c = {}
    for k, c in enumerate(clusters):
        for cid in c: c2c[cid] = k

    print(f"  [RandomCluster] Initial clusters: {clusters}")

    for rnd in range(warmup + 1, total_rounds + 1):
        t0 = time.time()

        if R > 0 and (rnd - warmup) % R == 0:
            clusters = random_partition(N, K, config["seed"] + rnd)
            for k, c in enumerate(clusters):
                for cid in c: c2c[cid] = k % len(cluster_models)

        cu = {k: [] for k in range(len(cluster_models))}
        cw = {k: [] for k in range(len(cluster_models))}

        for i in range(N):
            k  = c2c[i]
            lm = copy.deepcopy(cluster_models[k])
            lm = local_train_adult_standard(lm, client_train_loaders[i],
                config["local_epochs_base"], config["lr"],
                config["weight_decay"], device)
            cu[k].append(lm)
            cw[k].append(sizes[i])

        for k in cu:
            if cu[k]:
                tw  = sum(cw[k])
                agg = fedavg_aggregate(cu[k], [w / tw for w in cw[k]])
                for p, ap in zip(cluster_models[k].parameters(), agg):
                    p.data.copy_(ap)

        cs = [sum(sizes[i] for i in range(N) if c2c.get(i) == k)
              for k in range(len(cluster_models))]
        ts = sum(s for s in cs if s > 0)
        vm = [cluster_models[k] for k, s in enumerate(cs) if s > 0]
        vw = [s / ts for s in cs if s > 0]
        if vm:
            agg = fedavg_aggregate(vm, vw)
            for p, ap in zip(global_model.parameters(), agg):
                p.data.copy_(ap)

        g_acc  = evaluate_model(global_model, global_test_loader, device)
        c_accs = [evaluate_model(cluster_models[c2c[i]], client_val_loaders[i], device)
                  for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time() - t0)
        if rnd % 10 == 0 or rnd == warmup + 1:
            print_round_summary("RandomCluster", rnd, total_rounds,
                                 g_acc, sum(c_accs) / N)

    # Safe summary print
    if tracker.global_accs:
        print(f"  [RandomCluster] Best global: {max(tracker.global_accs)*100:.2f}%  "
              f"Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    else:
        print("  [RandomCluster] No rounds completed (warmup >= total_rounds)")
    return tracker
