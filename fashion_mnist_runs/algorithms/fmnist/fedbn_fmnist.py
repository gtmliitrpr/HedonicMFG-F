"""FedBN — FashionMNIST. No BN layers in FashionCNN → standard FedAvg."""
import copy, time
from utils import ResultsTracker, print_round_summary, evaluate_model, fedavg_aggregate
from local_trainer_fmnist import local_train_fmnist_fedbn
from models_fmnist import get_fmnist_model

def run_fedbn_fmnist(config, client_train_loaders, client_val_loaders,
                     global_test_loader, device):
    print("\n" + "="*50 + "\n  Running: FedBN [FashionMNIST]\n" + "="*50)
    print("  Note: FashionCNN has no BN — FedBN equivalent to FedAvg")
    tracker = ResultsTracker("FedBN")
    N = config["num_clients"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    total = sum(sizes)
    global_model = get_fmnist_model(config).to(device)

    for rnd in range(1, config["total_rounds"] + 1):
        t0 = time.time()
        local_models = []
        for i in range(N):
            lm = copy.deepcopy(global_model)
            lm = local_train_fmnist_fedbn(lm, client_train_loaders[i],
                config["local_epochs_base"], config["lr"],
                config["momentum"], config["weight_decay"], device)
            local_models.append(lm)
        agg = fedavg_aggregate(local_models, [s/total for s in sizes])
        for p, ap in zip(global_model.parameters(), agg): p.data.copy_(ap)
        g_acc  = evaluate_model(global_model, global_test_loader, device)
        c_accs = [evaluate_model(global_model, client_val_loaders[i], device) for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time()-t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("FedBN", rnd, config["total_rounds"], g_acc, sum(c_accs)/N)

    print(f"  [FedBN] Best global: {max(tracker.global_accs)*100:.2f}%  Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
