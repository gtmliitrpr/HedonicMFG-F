"""
local_trainer_adult.py — Local training routines for Adult Census (tabular MLP).

Key differences from MNIST version:
  - Adam optimizer instead of SGD (better for tabular MLP)
  - Weighted cross-entropy to handle class imbalance (income skew)
  - FedBN uses layer-index-based BN detection for Sequential modules
  - HedonicMFG personal head is 2-layer (not linear) for tabular
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np


def get_class_weights(dataloader, device, num_classes=2):
    """Compute inverse frequency class weights for imbalanced binary labels."""
    counts = torch.zeros(num_classes)
    for _, y in dataloader:
        for c in range(num_classes):
            counts[c] += (y == c).sum()
    total = counts.sum()
    weights = total / (num_classes * counts + 1e-8)
    return weights.to(device)


# ──────────────────────────────────────────
# Standard local training (Adam)
# ──────────────────────────────────────────
def local_train_adult_standard(model, dataloader, epochs, lr,
                                weight_decay, device,
                                participation_rate=1.0,
                                use_weighted_loss=True):
    """Standard local Adam training for tabular MLP."""
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    if use_weighted_loss:
        class_weights = get_class_weights(dataloader, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for x, y in dataloader:
            if participation_rate < 1.0:
                if np.random.random() > participation_rate:
                    continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return model


# ──────────────────────────────────────────
# FedProx
# ──────────────────────────────────────────
def local_train_adult_fedprox(model, global_model, dataloader, epochs,
                               lr, weight_decay, mu, device):
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = get_class_weights(dataloader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    global_params = [p.data.clone() for p in global_model.parameters()]

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            prox = sum((p - gp).norm() ** 2
                       for p, gp in zip(model.parameters(), global_params))
            loss += (mu / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return model


# ──────────────────────────────────────────
# SCAFFOLD
# ──────────────────────────────────────────
def local_train_adult_scaffold(model, global_model, dataloader, epochs,
                                lr, c_i, c_global, device):
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    c_i_dev  = [c.to(device) for c in c_i]
    c_g_dev  = [c.to(device) for c in c_global]
    init_params = [p.data.clone() for p in model.parameters()]
    num_steps = 0

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            for p, ci, cg in zip(model.parameters(), c_i_dev, c_g_dev):
                if p.grad is not None:
                    p.grad.data -= ci - cg
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            num_steps += 1

    if num_steps == 0:
        num_steps = 1
    new_c_i, c_delta = [], []
    for ci, cg, ip, p in zip(c_i_dev, c_g_dev, init_params, model.parameters()):
        new_ci = ci - cg + (ip.to(device) - p.data) / (num_steps * lr)
        new_c_i.append(new_ci.cpu())
        c_delta.append((new_ci - ci).cpu())

    return model, new_c_i, c_delta


# ──────────────────────────────────────────
# MOON
# ──────────────────────────────────────────
def local_train_adult_moon(model, global_model, prev_model, dataloader,
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
    cos_sim = nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
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

            loss = ce_loss + mu * con_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    return model


# ──────────────────────────────────────────
# FedBN — training identical, BN stays local
# ──────────────────────────────────────────
def local_train_adult_fedbn(model, dataloader, epochs, lr,
                             weight_decay, device):
    return local_train_adult_standard(model, dataloader, epochs, lr,
                                       weight_decay, device)


# ──────────────────────────────────────────
# pFedME
# ──────────────────────────────────────────
def local_train_adult_pfedme(model, global_model, dataloader, local_steps,
                              lr, beta, lambda_reg, device):
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()

    personal_model = copy.deepcopy(global_model).to(device)
    personal_model.train()
    p_optimizer = optim.Adam(personal_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    global_params = [p.data.clone() for p in global_model.parameters()]

    for step in range(local_steps):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            p_optimizer.zero_grad()
            out = personal_model(x)
            loss = criterion(out, y)
            prox = sum((pp - gp).norm() ** 2
                       for pp, gp in zip(personal_model.parameters(), global_params))
            loss += (lambda_reg / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(personal_model.parameters(), max_norm=5.0)
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
def local_train_adult_hedonic(model, dataloader, epochs, lr,
                               weight_decay, device,
                               participation_rate=1.0,
                               personal_head=None,
                               finetune_epochs=3):
    """
    HedonicMFG local training for Adult Census:
    1. Train backbone with MFG-determined (epochs, participation_rate)
    2. Fine-tune 2-layer personal head on full local data (backbone frozen)
    """
    # Step 1: backbone training
    model = local_train_adult_standard(
        model, dataloader, epochs, lr, weight_decay,
        device, participation_rate
    )

    # Step 2: personal head fine-tuning
    if personal_head is not None:
        model.eval()
        personal_head = personal_head.to(device)
        personal_head.train()
        head_opt = optim.Adam(personal_head.parameters(), lr=lr * 3)
        criterion = nn.CrossEntropyLoss()

        for _ in range(finetune_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                head_opt.zero_grad()
                with torch.no_grad():
                    feat = model.get_features(x)
                out = personal_head(feat)
                loss = criterion(out, y)
                loss.backward()
                head_opt.step()

    return model, personal_head
