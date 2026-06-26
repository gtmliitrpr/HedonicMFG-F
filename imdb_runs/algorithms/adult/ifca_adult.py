"""IFCA — Adult Census"""
import torch, copy, time, numpy as np
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from local_trainer_adult import local_train_adult_standard
from models_adult import get_adult_model

def run_ifca_adult(config, client_train_loaders, client_val_loaders,
                   global_test_loader, device, feature_dim):
    print("\n" + "="*50 + "\n  Running: IFCA [Adult]\n" + "="*50)
    tracker = ResultsTracker("IFCA")
    N = config["num_clients"]
    K = config["ifca_num_clusters"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    cluster_models = [get_adult_model(config, feature_dim).to(device) for _ in range(K)]

    def est_loss(model, loader):
        model.eval()
        crit = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in loader:
                return crit(model(x.to(device)), y.to(device)).item()
        return 1e9

    for rnd in range(1, config["total_rounds"] + 1):
        t0 = time.time()
        assignments = [int(np.argmin([est_loss(cluster_models[k], client_train_loaders[i]) for k in range(K)])) for i in range(N)]
        updates = {k: [] for k in range(K)}
        wts    = {k: [] for k in range(K)}

        for i in range(N):
            k = assignments[i]
            lm = copy.deepcopy(cluster_models[k])
            lm = local_train_adult_standard(lm, client_train_loaders[i],
                config["local_epochs_base"], config["lr"],
                config["weight_decay"], device)
            updates[k].append(lm); wts[k].append(sizes[i])

        for k in range(K):
            if updates[k]:
                tw = sum(wts[k])
                agg = fedavg_aggregate(updates[k], [w/tw for w in wts[k]])
                for p, ap in zip(cluster_models[k].parameters(), agg): p.data.copy_(ap)

        cluster_sizes = [sum(wts[k]) for k in range(K)]
        total_s = sum(cluster_sizes)
        global_eval = get_adult_model(config, feature_dim).to(device)
        agg = fedavg_aggregate(cluster_models, [s/total_s if total_s>0 else 1/K for s in cluster_sizes])
        for p, ap in zip(global_eval.parameters(), agg): p.data.copy_(ap)

        g_acc = evaluate_model(global_eval, global_test_loader, device)
        c_accs = [evaluate_model(cluster_models[assignments[i]], client_val_loaders[i], device) for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time()-t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("IFCA", rnd, config["total_rounds"], g_acc, sum(c_accs)/N)

    print(f"  [IFCA] Best global: {max(tracker.global_accs)*100:.2f}%  Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
