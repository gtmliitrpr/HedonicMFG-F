"""FedBN — Adult Census. BN layers stay local per client."""
import copy, time
from utils import ResultsTracker, print_round_summary, evaluate_model
from local_trainer_adult import local_train_adult_fedbn
from models_adult import get_adult_model, aggregate_except_bn_adult

def run_fedbn_adult(config, client_train_loaders, client_val_loaders,
                    global_test_loader, device, feature_dim):
    print("\n" + "="*50 + "\n  Running: FedBN [Adult]\n" + "="*50)
    tracker = ResultsTracker("FedBN")
    N = config["num_clients"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    total = sum(sizes)
    global_model = get_adult_model(config, feature_dim).to(device)
    local_models = [copy.deepcopy(global_model) for _ in range(N)]

    for rnd in range(1, config["total_rounds"] + 1):
        t0 = time.time()
        # Distribute global non-BN params
        g_state = global_model.state_dict()
        for i in range(N):
            c_state = local_models[i].state_dict()
            for key in g_state:
                parts = key.split(".")
                is_bn = len(parts) >= 2 and parts[1] == "1"
                if not is_bn:
                    c_state[key] = g_state[key].clone()
            local_models[i].load_state_dict(c_state)

        updated = []
        for i in range(N):
            trained = local_train_adult_fedbn(local_models[i],
                client_train_loaders[i], config["local_epochs_base"],
                config["lr"], config["weight_decay"], device)
            local_models[i] = trained
            updated.append(trained)

        agg_state = aggregate_except_bn_adult(updated, sizes)
        g_state = global_model.state_dict()
        g_state.update(agg_state)
        global_model.load_state_dict(g_state)

        g_acc = evaluate_model(global_model, global_test_loader, device)
        c_accs = [evaluate_model(local_models[i], client_val_loaders[i], device) for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time()-t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("FedBN", rnd, config["total_rounds"], g_acc, sum(c_accs)/N)

    print(f"  [FedBN] Best global: {max(tracker.global_accs)*100:.2f}%  Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
