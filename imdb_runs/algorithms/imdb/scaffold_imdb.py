"""SCAFFOLD — IMDB. Control variates on CPU to avoid device mismatch."""
import torch, copy, time
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from local_trainer_imdb import local_train_imdb_scaffold
from models_imdb import get_imdb_model

def run_scaffold_imdb(config, client_train_loaders, client_val_loaders,
                      global_test_loader, device, vocab_size):
    print("\n" + "="*50 + "\n  Running: SCAFFOLD [IMDB]\n" + "="*50)
    tracker = ResultsTracker("SCAFFOLD")
    N = config["num_clients"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    total = sum(sizes)
    lr = config["scaffold_lr"]
    global_model = get_imdb_model(config, vocab_size).to(device)

    # All control variates on CPU
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
            lm, new_ci, c_delta = local_train_imdb_scaffold(
                lm, global_model, client_train_loaders[i],
                config["local_epochs_base"], lr, c_clients[i], c_global, device)
            c_clients[i] = new_ci
            local_models.append(lm)
            c_deltas.append(c_delta)

        weights = [s/total for s in sizes]
        agg = fedavg_aggregate(local_models, weights)
        for p, ap in zip(global_model.parameters(), agg): p.data.copy_(ap.to(device))

        for idx, c_g in enumerate(c_global):
            delta_sum = sum(c_deltas[i][idx] for i in range(N))
            c_g.data.add_(delta_sum / N)

        g_acc = evaluate_model(global_model, global_test_loader, device)
        c_accs = [evaluate_model(global_model, client_val_loaders[i], device) for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time()-t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("SCAFFOLD", rnd, config["total_rounds"], g_acc, sum(c_accs)/N)

    print(f"  [SCAFFOLD] Best global: {max(tracker.global_accs)*100:.2f}%  Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
