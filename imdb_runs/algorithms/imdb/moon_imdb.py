"""MOON — IMDB"""
import copy, time
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from local_trainer_imdb import local_train_imdb_moon
from models_imdb import get_imdb_model

def run_moon_imdb(config, client_train_loaders, client_val_loaders,
                  global_test_loader, device, vocab_size):
    print("\n" + "="*50 + "\n  Running: MOON [IMDB]\n" + "="*50)
    tracker = ResultsTracker("MOON")
    N = config["num_clients"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    total = sum(sizes)
    global_model = get_imdb_model(config, vocab_size).to(device)
    prev_models = [copy.deepcopy(global_model) for _ in range(N)]

    for rnd in range(1, config["total_rounds"] + 1):
        t0 = time.time()
        local_models = []
        for i in range(N):
            lm = copy.deepcopy(global_model)
            lm = local_train_imdb_moon(lm, global_model, prev_models[i],
                client_train_loaders[i], config["local_epochs_base"],
                config["lr"], config["weight_decay"],
                config["moon_mu"], config["moon_temperature"], device)
            local_models.append(lm)
        prev_models = [copy.deepcopy(m) for m in local_models]
        agg = fedavg_aggregate(local_models, [s/total for s in sizes])
        for p, ap in zip(global_model.parameters(), agg): p.data.copy_(ap)
        g_acc = evaluate_model(global_model, global_test_loader, device)
        c_accs = [evaluate_model(global_model, client_val_loaders[i], device) for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time()-t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("MOON", rnd, config["total_rounds"], g_acc, sum(c_accs)/N)

    print(f"  [MOON] Best global: {max(tracker.global_accs)*100:.2f}%  Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
