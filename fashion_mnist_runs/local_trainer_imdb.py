"""
local_trainer_imdb.py — Local training routines for IMDB (TextCNN).

Key differences from MNIST/Adult:
  - Adam optimizer (better for NLP embeddings)
  - Gradient clipping at 1.0 (NLP standard)
  - No weighted CE — IMDB classes are balanced overall
  - MOON contrastive loss on text features
  - HedonicMFG tracks individual client models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np


# ──────────────────────────────────────────
# Standard local training (Adam)
# ──────────────────────────────────────────
def local_train_imdb_standard(model, dataloader, epochs, lr,
                               weight_decay, device):
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


# ──────────────────────────────────────────
# FedProx
# ──────────────────────────────────────────
def local_train_imdb_fedprox(model, global_model, dataloader, epochs,
                              lr, weight_decay, mu, device):
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    global_params = [p.data.clone() for p in global_model.parameters()]

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            prox = sum((p - gp).norm() ** 2
                       for p, gp in zip(model.parameters(), global_params))
            loss += (mu / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


# ──────────────────────────────────────────
# SCAFFOLD
# ──────────────────────────────────────────
def local_train_imdb_scaffold(model, global_model, dataloader, epochs,
                               lr, c_i, c_global, device):
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    c_i_dev = [c.to(device) for c in c_i]
    c_g_dev = [c.to(device) for c in c_global]
    init_params = [p.data.clone() for p in model.parameters()]
    num_steps = 0

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            for p, ci, cg in zip(model.parameters(), c_i_dev, c_g_dev):
                if p.grad is not None:
                    p.grad.data.add_(cg - ci)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            num_steps += 1

    if num_steps == 0:
        num_steps = 1

    new_c_i, c_delta = [], []
    for ci_d, cg_d, ip, p in zip(c_i_dev, c_g_dev, init_params, model.parameters()):
        new_ci = ci_d - cg_d + (ip.to(device) - p.data) / (num_steps * lr)
        new_c_i.append(new_ci.cpu())
        c_delta.append((new_ci - ci_d).cpu())

    return model, new_c_i, c_delta


# ──────────────────────────────────────────
# MOON
# ──────────────────────────────────────────
def local_train_imdb_moon(model, global_model, prev_model, dataloader,
                           epochs, lr, weight_decay, mu, temperature, device):
    model = model.to(device)
    global_model = global_model.to(device)
    global_model.eval()
    if prev_model is not None:
        prev_model = prev_model.to(device)
        prev_model.eval()

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    cos_sim   = nn.CosineSimilarity(dim=-1)

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, z_local = model.forward_with_features(x)
            ce_loss = criterion(logits, y)

            with torch.no_grad():
                _, z_global = global_model.forward_with_features(x)
                if prev_model is not None:
                    _, z_prev = prev_model.forward_with_features(x)

            if prev_model is not None:
                pos = cos_sim(z_local, z_global) / temperature
                neg = cos_sim(z_local, z_prev) / temperature
                con_loss = -torch.log(
                    torch.exp(pos) / (torch.exp(pos) + torch.exp(neg) + 1e-8)
                ).mean()
            else:
                con_loss = torch.tensor(0.0, device=device)

            (ce_loss + mu * con_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


# ──────────────────────────────────────────
# FedBN (TextCNN has no BN — same as standard)
# ──────────────────────────────────────────
def local_train_imdb_fedbn(model, dataloader, epochs, lr,
                            weight_decay, device):
    return local_train_imdb_standard(model, dataloader, epochs, lr,
                                      weight_decay, device)


# ──────────────────────────────────────────
# pFedME
# ──────────────────────────────────────────
def local_train_imdb_pfedme(model, global_model, dataloader, local_steps,
                             lr, beta, lambda_reg, device):
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()

    personal_model = copy.deepcopy(global_model).to(device)
    personal_model.train()
    p_optimizer = optim.Adam(personal_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    global_params = [p.data.clone() for p in global_model.parameters()]

    for _ in range(local_steps):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            p_optimizer.zero_grad()
            out = personal_model(x)
            loss = criterion(out, y)
            prox = sum((pp - gp).norm() ** 2
                       for pp, gp in zip(personal_model.parameters(), global_params))
            loss += (lambda_reg / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(personal_model.parameters(), max_norm=1.0)
            p_optimizer.step()
            break

    with torch.no_grad():
        for gp, pp, orig_gp in zip(model.parameters(),
                                    personal_model.parameters(),
                                    global_params):
            gp.data = orig_gp - beta * lambda_reg * (orig_gp - pp.data)

    return model, personal_model


# ──────────────────────────────────────────
# HedonicMFG local training
# ──────────────────────────────────────────
def local_train_imdb_hedonic(model, dataloader, epochs, lr,
                              weight_decay, device,
                              personal_head=None, finetune_epochs=3):
    """
    HedonicMFG local training for IMDB.
    Step 1: Train backbone with Adam.
    Step 2: Fine-tune personal head (backbone frozen).
    """
    model = local_train_imdb_standard(model, dataloader, epochs, lr,
                                       weight_decay, device)

    if personal_head is not None:
        model.eval()
        personal_head = personal_head.to(device)
        personal_head.train()
        head_opt = optim.Adam(personal_head.parameters(), lr=lr * 0.5)
        criterion = nn.CrossEntropyLoss()

        for _ in range(finetune_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                head_opt.zero_grad()
                with torch.no_grad():
                    feat = model.get_features(x)
                criterion(personal_head(feat), y).backward()
                head_opt.step()

    return model, personal_head
