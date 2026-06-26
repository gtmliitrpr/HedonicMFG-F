"""pFedME — IMDB"""
import copy, time
from utils import ResultsTracker, print_round_summary, evaluate_model
from local_trainer_imdb import local_train_imdb_pfedme
from models_imdb import get_imdb_model

def run_pfedme_imdb(config, client_train_loaders, client_val_loaders,
                    global_test_loader, device, vocab_size):
    print("\n" + "="*50 + "\n  Running: pFedME [IMDB]\n" + "="*50)
    tracker = ResultsTracker("pFedME")
    N = config["num_clients"]
    sizes = [len(l.dataset) for l in client_train_loaders]
    total = sum(sizes)
    global_model = get_imdb_model(config, vocab_size).to(device)
    personal_models = [copy.deepcopy(global_model) for _ in range(N)]

    for rnd in range(1, config["total_rounds"] + 1):
        t0 = time.time()
        updated_globals = []
        for i in range(N):
            w_i = copy.deepcopy(global_model)
            w_i, personal_models[i] = local_train_imdb_pfedme(
                w_i, global_model, client_train_loaders[i],
                config["pfedme_local_steps"], config["lr"],
                config["pfedme_beta"], config["pfedme_lambda"], device)
            updated_globals.append(w_i)

        norm_w = [s/total for s in sizes]
        new_state = {}
        for key in global_model.state_dict():
            new_state[key] = sum(w * m.state_dict()[key].float()
                                  for w, m in zip(norm_w, updated_globals))
        global_model.load_state_dict(new_state)

        g_acc = evaluate_model(global_model, global_test_loader, device)
        c_accs = [evaluate_model(personal_models[i], client_val_loaders[i], device) for i in range(N)]
        tracker.log(rnd, g_acc, c_accs, time.time()-t0)
        if rnd % 10 == 0 or rnd == 1:
            print_round_summary("pFedME", rnd, config["total_rounds"], g_acc, sum(c_accs)/N)

    print(f"  [pFedME] Best global: {max(tracker.global_accs)*100:.2f}%  Best client: {max(tracker.avg_client_accs)*100:.2f}%")
    return tracker
