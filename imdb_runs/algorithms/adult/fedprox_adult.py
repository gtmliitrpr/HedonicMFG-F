"""FedProx — Adult Census"""
import copy, time
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from local_trainer_adult import local_train_adult_fedprox
from models_adult import get_adult_model

def run_fedprox_adult(config, client_train_loaders, client_val_loaders,
                      global_test_loader, device, feature_dim):
    print("\n" + "="*50 + "\n  Running: FedProx [Adult]\n" + "="*50)
    tracker = ResultsTracker("FedProx")
    N = config["num_clients"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    total = sum(sizes)
    global_model = get_adult_model(config, feature_dim).to(device)

    for rnd in range(1, config["total_rounds"] + 1):
        t0 = time.time()
        local_models = []
        for i in range(N):
            lm = copy.deepcopy(global_model)
            lm = local_train_adult_fedprox(lm, global_model,
                client_train_loaders[i], config["local_epochs_base"],
                config["lr"], config["weight_decay"], config["fedprox_mu"], device)
            local_models.append(lm)
        agg = fedavg_aggregate(local_models, [s/total for s in sizes])
        for p, ap in zip(global_model.parameters(), agg): p.data.copy_(ap)
        g_acc = evaluate_model(global_model, global_test_loader, device)
        c_accs = [evaluate_model(global_model, client_val_loaders[i], device) for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time()-t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("FedProx", rnd, config["total_rounds"], g_acc, sum(c_accs)/N)

    print(f"  [FedProx] Best global: {max(tracker.global_accs)*100:.2f}%  Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
