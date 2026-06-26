"""
local_trainer.py — Local training routines used by all FL algorithms.
Each function returns updated model params and optionally gradients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from utils import get_gradients


# ──────────────────────────────────────────
# Standard local SGD
# ──────────────────────────────────────────
def local_train_standard(model, dataloader, epochs, lr, momentum,
                          weight_decay, device, participation_rate=1.0):
    """
    Standard local SGD training (used by FedAvg, IFCA, CFL, Random Clustering).
    participation_rate: fraction of local data to use (MFG action τ).
    Returns updated model (in-place).
    """
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            # Apply participation rate: skip some batches
            if participation_rate < 1.0:
                if np.random.random() > participation_rate:
                    continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    return model


# ──────────────────────────────────────────
# FedProx local training
# ──────────────────────────────────────────
def local_train_fedprox(model, global_model, dataloader, epochs, lr,
                         momentum, weight_decay, mu, device):
    """
    FedProx: adds proximal term μ/2 ||w - w_global||² to loss.
    From: Li et al. 2020, 'Federated Optimization in Heterogeneous Networks'
    """
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    global_params = [p.data.clone() for p in global_model.parameters()]

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            # Proximal term
            prox = 0.0
            for p, gp in zip(model.parameters(), global_params):
                prox += (p - gp).norm() ** 2
            loss += (mu / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    return model


# ──────────────────────────────────────────
# SCAFFOLD local training
# ──────────────────────────────────────────
def local_train_scaffold(model, global_model, dataloader, epochs, lr,
                          c_i, c_global, device):
    """
    SCAFFOLD: corrects client drift via control variates.
    From: Karimireddy et al. 2020, 'SCAFFOLD: Stochastic Controlled Averaging'
    c_i      : client control variate (list of tensors)
    c_global : global control variate (list of tensors)
    Returns  : (updated_model, new_c_i, c_delta)
    """
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    # Move control variates to device
    c_i_dev = [c.to(device) for c in c_i]
    c_g_dev = [c.to(device) for c in c_global]

    init_params = [p.data.clone() for p in model.parameters()]
    num_steps = 0

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            # Apply SCAFFOLD correction: g = g - c_i + c_global
            for p, ci, cg in zip(model.parameters(), c_i_dev, c_g_dev):
                if p.grad is not None:
                    p.grad.data -= ci - cg
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            num_steps += 1

    # Update client control variate
    # c_i^+ = c_i - c_global + (x_0 - x) / (K * lr)
    if num_steps == 0:
        num_steps = 1
    new_c_i = []
    c_delta = []
    for ci, cg, ip, p in zip(c_i_dev, c_g_dev, init_params, model.parameters()):
        new_ci = ci - cg + (ip.to(device) - p.data) / (num_steps * lr)
        new_c_i.append(new_ci.cpu())
        c_delta.append((new_ci - ci).cpu())

    return model, new_c_i, c_delta


# ──────────────────────────────────────────
# MOON local training
# ──────────────────────────────────────────
def local_train_moon(model, global_model, prev_model, dataloader,
                      epochs, lr, momentum, weight_decay, mu,
                      temperature, device):
    """
    MOON: Model-Contrastive Federated Learning.
    From: Li et al. 2021, 'Model-Contrastive Federated Learning'
    Uses contrastive loss between local, global, and previous local representations.
    """
    model = model.to(device)
    global_model = global_model.to(device)
    global_model.eval()
    if prev_model is not None:
        prev_model = prev_model.to(device)
        prev_model.eval()

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Local representation
            logits, z_local = model.forward_with_features(x)
            ce_loss = criterion(logits, y)

            # Contrastive loss
            with torch.no_grad():
                _, z_global = global_model.forward_with_features(x)
                if prev_model is not None:
                    _, z_prev = prev_model.forward_with_features(x)

            if prev_model is not None:
                pos = cos_sim(z_local, z_global) / temperature
                neg = cos_sim(z_local, z_prev) / temperature
                # MOON contrastive: maximize similarity to global, minimize to prev
                con_loss = -torch.log(
                    torch.exp(pos) / (torch.exp(pos) + torch.exp(neg) + 1e-8)
                ).mean()
            else:
                con_loss = torch.tensor(0.0, device=device)

            loss = ce_loss + mu * con_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    return model


# ──────────────────────────────────────────
# FedBN local training
# ──────────────────────────────────────────
def local_train_fedbn(model, dataloader, epochs, lr, momentum,
                       weight_decay, device):
    """
    FedBN: standard SGD but BN layers stay local (handled at aggregation).
    From: Li et al. 2021, 'FedBN: Federated Learning on Non-IID Features via Local BN'
    Training is identical to standard — BN locality enforced at aggregation.
    """
    return local_train_standard(model, dataloader, epochs, lr,
                                 momentum, weight_decay, device)


# ──────────────────────────────────────────
# pFedME local training
# ──────────────────────────────────────────
def local_train_pfedme(model, global_model, dataloader, local_steps,
                        lr, beta, lambda_reg, device):
    """
    pFedME: Personalized Federated Learning via Moreau Envelope.
    From: Dinh et al. 2020, 'Personalized Federated Learning with Moreau Envelopes'

    Alternates between:
      1. θ_i update: minimizes f_i(θ) + λ/2 ||θ - w||² (personalized model)
      2. w update: gradient step toward θ_i
    """
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()

    # Personal model starts from global
    personal_model = copy.deepcopy(global_model).to(device)
    personal_model.train()
    p_optimizer = optim.SGD(personal_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    global_params = [p.data.clone() for p in global_model.parameters()]

    for step in range(local_steps):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            p_optimizer.zero_grad()
            out = personal_model(x)
            loss = criterion(out, y)
            # Moreau envelope proximal term
            prox = sum(
                (pp - gp).norm() ** 2
                for pp, gp in zip(personal_model.parameters(), global_params)
            )
            loss += (lambda_reg / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(personal_model.parameters(), max_norm=10.0)
            p_optimizer.step()
            break  # one batch per step for efficiency

    # Update global model w via gradient step: w = w - beta * lambda * (w - θ_i)
    with torch.no_grad():
        for gp, pp, orig_gp in zip(model.parameters(),
                                    personal_model.parameters(),
                                    global_params):
            gp.data = orig_gp - beta * lambda_reg * (orig_gp - pp.data)

    return model, personal_model


# ──────────────────────────────────────────
# HedonicMFG local training (with personalized head fine-tuning)
# ──────────────────────────────────────────
def local_train_hedonic(model, dataloader, epochs, lr, momentum,
                         weight_decay, device, participation_rate=1.0,
                         personal_head=None, finetune_epochs=2):
    """
    HedonicMFG local training:
    1. Train shared backbone with MFG-determined (epochs, participation_rate).
    2. If personal_head provided, fine-tune it on full local data.
    """
    # Step 1: Backbone training with MFG strategy
    model = local_train_standard(model, dataloader, epochs, lr,
                                  momentum, weight_decay, device,
                                  participation_rate)

    # Step 2: Fine-tune personalized head (backbone frozen)
    if personal_head is not None:
        model.eval()
        personal_head = personal_head.to(device)
        personal_head.train()
        head_optimizer = optim.Adam(personal_head.parameters(), lr=lr * 2)
        criterion = nn.CrossEntropyLoss()

        for _ in range(finetune_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                head_optimizer.zero_grad()
                with torch.no_grad():
                    feat = model.get_features(x)
                out = personal_head(feat)
                loss = criterion(out, y)
                loss.backward()
                head_optimizer.step()

    return model, personal_head
