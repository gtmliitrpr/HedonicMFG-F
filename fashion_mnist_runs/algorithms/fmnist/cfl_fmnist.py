"""CFL — FashionMNIST"""
import torch, copy, time, numpy as np
from utils import (ResultsTracker, print_round_summary, evaluate_model,
                    fedavg_aggregate, get_model_params, pairwise_cosine_similarity)
from local_trainer_fmnist import local_train_fmnist_standard
from models_fmnist import get_fmnist_model

def run_cfl_fmnist(config, client_train_loaders, client_val_loaders,
                   global_test_loader, device):
    print("\n" + "="*50 + "\n  Running: CFL [FashionMNIST]\n" + "="*50)
    tracker = ResultsTracker("CFL")
    N = config["num_clients"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    eps1, eps2 = config["cfl_eps1"], config["cfl_eps2"]

    clusters = [list(range(N))]
    cluster_models = {0: get_fmnist_model(config).to(device)}
    c2c = {i: 0 for i in range(N)}
    next_id = 1

    for rnd in range(1, config["total_rounds"] + 1):
        t0 = time.time()
        all_trained, all_updates = {}, {}

        for clist in list(clusters):
            cid   = c2c[clist[0]]
            model = cluster_models[cid]
            init_p = get_model_params(model)
            for i in clist:
                lm = copy.deepcopy(model)
                lm = local_train_fmnist_standard(lm, client_train_loaders[i],
                    config["local_epochs_base"], config["lr"],
                    config["momentum"], config["weight_decay"], device)
                all_trained[i] = lm
                lp = get_model_params(lm)
                all_updates[i] = torch.cat([(lp2-ip).flatten() for lp2,ip in zip(lp,init_p)])

            if rnd > 5 and len(clist) >= 2:
                vecs = [all_updates[i] for i in clist]
                sim_matrix = pairwise_cosine_similarity(vecs)
                norms = [v.norm().item() for v in vecs]
                if sim_matrix.mean() < eps1 and (max(norms)-min(norms)) > eps2:
                    mean_norm = np.mean(norms)
                    ga = [c for c,n in zip(clist,norms) if n >= mean_norm]
                    gb = [c for c,n in zip(clist,norms) if n < mean_norm]
                    if ga and gb:
                        new_id = next_id; next_id += 1
                        cluster_models[new_id] = copy.deepcopy(cluster_models[cid])
                        clusters.remove(clist); clusters.append(ga); clusters.append(gb)
                        for c in ga: c2c[c] = cid
                        for c in gb: c2c[c] = new_id

        for clist in list(clusters):
            cid = c2c[clist[0]]
            ms = [all_trained[i] for i in clist if i in all_trained]
            ws = [sizes[i] for i in clist if i in all_trained]
            if ms:
                tw = sum(ws)
                agg = fedavg_aggregate(ms, [w/tw for w in ws])
                for p, ap in zip(cluster_models[cid].parameters(), agg): p.data.copy_(ap)

        all_cids = list({c2c[i] for i in range(N)})
        cpop = {cid: sum(sizes[i] for i in range(N) if c2c[i]==cid) for cid in all_cids}
        total_p = sum(cpop.values())
        global_eval = get_fmnist_model(config).to(device)
        agg = fedavg_aggregate([cluster_models[cid] for cid in all_cids],
                                [cpop[cid]/total_p for cid in all_cids])
        for p, ap in zip(global_eval.parameters(), agg): p.data.copy_(ap)

        g_acc  = evaluate_model(global_eval, global_test_loader, device)
        c_accs = [evaluate_model(cluster_models[c2c[i]], client_val_loaders[i], device)
                  for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time()-t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("CFL", rnd, config["total_rounds"], g_acc, sum(c_accs)/N)

    print(f"  [CFL] Best global: {max(tracker.global_accs)*100:.2f}%  Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
