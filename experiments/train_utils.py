import os
import numpy as np
import random
import copy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score

from FeatureAcquisition import FeatureAcquisition


# 임의로 masking하는 함수
def sample_mask_uniform_K_per_sample(bs, d, min_K, max_K): # batch size, feature 개수, 최소 관측 샘플 수, 최대 관측 샘플 수
    m = np.zeros((bs, d), dtype=np.float32)
    Ks = np.random.randint(min_K, max_K+1, size=(bs,))
    for i, K in enumerate(Ks): # Ks의 index와 해당 index의 값
        idx = np.random.choice(d, size=K, replace=False)
        m[i, idx] = 1.0
    return m


# 시드 고정
def set_seed(numpy_seed, random_seed, torch_seed_cpu, torch_seed_cuda):
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed_cpu)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed_cuda)
        torch.cuda.manual_seed_all(torch_seed_cuda)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# predictor trainer
def train_predictor(
    predictor,
    train_loader,
    X_val,
    y_val,
    D,
    epochs,
    optimizer,
    criterion,
    lr_factor=0.2,
    cooldown=0,
    min_lr=1e-7,
    scheduler_patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_factor,
        patience=scheduler_patience,
        cooldown=cooldown,
        min_lr=min_lr,
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train 
        predictor.train()
        total_loss = 0.0
        total_train_samples = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            m_np = sample_mask_uniform_K_per_sample(
                bs=xb.size(0),
                d=D,
                min_K=1,
                max_K=D
            )
            mb = torch.tensor(m_np, dtype=torch.float32, device=device)

            logits = predictor(xb, mb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_train_samples += xb.size(0)

        avg_train_loss = total_loss / total_train_samples

        # Validation 
        predictor.eval()
        with torch.no_grad():
            m_np = sample_mask_uniform_K_per_sample(
                bs=X_val.size(0),
                d=D,
                min_K=1,
                max_K=D
            )
            mv = torch.tensor(m_np, dtype=torch.float32, device=device)

            logits_val = predictor(X_val, mv)
            val_loss = criterion(logits_val, y_val).item()

            preds = logits_val.argmax(dim=-1)
            acc = (preds == y_val).float().mean().item()

        # Scheduler update
        scheduler.step(acc)

        # 로그 출력
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # Best model 저장
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(predictor.state_dict())

    # 모든 epoch 후 best 모델 복원
    if best_state is not None:
        predictor.load_state_dict(best_state)

    print(f"Best validation accuracy = {best_acc:.4f}")


# Partial VAE trainer
def train_VAE(
    generator,
    train_loader,
    X_val,
    D, # feature 개수
    epochs,
    optimizer,
    obs_sigma=0.2,
    lr_factor=0.2,
    cooldown=0,
    min_lr=1e-7,
    scheduler_patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    X_val = X_val.to(device)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min", # loss 기준이기 때문 
        factor=lr_factor,
        patience=scheduler_patience,
        cooldown=cooldown,
        min_lr=min_lr,
    )

    best_val_loss = float("inf")
    best_state = None

    for ep in range(epochs):
        # Train
        generator.train()
        total_loss, total_kl, total_nll = 0.0, 0.0, 0.0
        count = 0

        for xb, _ in train_loader:
            xb = xb.to(device).float()
            bs = xb.size(0)

            m_np = sample_mask_uniform_K_per_sample(
                bs=bs,
                d=D,
                min_K=1,
                max_K=D,
            )
            mb = torch.tensor(m_np, dtype=torch.float32, device=device)

            loss, logs = generator.loss_func(
                xb,
                mb,
                obs_sigma=obs_sigma,
                n_samples=1,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bs
            total_kl   += logs["KL"].item() * bs
            total_nll  += logs["NLL_X"].item() * bs
            count      += bs

        train_loss = total_loss / count
        train_kl   = total_kl / count
        train_nll  = total_nll / count

        # Validation
        generator.eval()
        with torch.no_grad():
            bs_val = X_val.size(0)
            m_np = sample_mask_uniform_K_per_sample(
                bs=bs_val,
                d=D,
                min_K=1,
                max_K=D,
            )
            mv = torch.tensor(m_np, dtype=torch.float32, device=device)

            val_loss_tensor, val_logs = generator.loss_func(
                X_val,
                mv,
                obs_sigma=obs_sigma,
                n_samples=1,
            )

            val_loss = val_loss_tensor.item()
            val_kl   = val_logs["KL"].item()
            val_nll  = val_logs["NLL_X"].item()

        # 스케줄러 업데이트 (val_loss 기준)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[PVAE ep {1 + ep:02d}] "
            f"train_loss={train_loss:.4f}  train_KL={train_kl:.4f}  train_NLL_X={train_nll:.4f} | "
            f"val_loss={val_loss:.4f}  val_KL={val_kl:.4f}  val_NLL_X={val_nll:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # best model 저장 (val_loss 최소 기준)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(generator.state_dict())

    # 가장 좋은 val_loss 기준으로 weight 복원
    if best_state is not None:
        generator.load_state_dict(best_state)

    print(f"Best val_loss = {best_val_loss:.4f}")    


# Flow trainer
def train_Flow(
    generator,
    train_loader,
    X_val,
    D,  # feature 개수
    epochs,
    optimizer,
    lr_factor=0.2,
    cooldown=0,
    min_lr=1e-7,
    scheduler_patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    X_val = X_val.to(device)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",  # loss 기준이기 때문 
        factor=lr_factor,
        patience=scheduler_patience,
        cooldown=cooldown,
        min_lr=min_lr,
    )

    best_val_loss = float("inf")
    best_state = None

    for ep in range(epochs):
        # Train
        generator.train()
        total_loss, total_nll = 0.0, 0.0
        count = 0

        for xb, _ in train_loader:
            xb = xb.to(device).float()
            bs = xb.size(0)

            m_np = sample_mask_uniform_K_per_sample(
                bs=bs,
                d=D,
                min_K=1,
                max_K=D,
            )
            mb = torch.tensor(m_np, dtype=torch.float32, device=device)

            loss, logs = generator.loss_func(
                xb,
                mb,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bs
            total_nll  += logs["NLL_xu"].item() * bs
            count      += bs

        train_loss = total_loss / count
        train_nll  = total_nll / count

        # Validation
        generator.eval()
        with torch.no_grad():
            bs_val = X_val.size(0)
            m_np = sample_mask_uniform_K_per_sample(
                bs=bs_val,
                d=D,
                min_K=1,
                max_K=D,
            )
            mv = torch.tensor(m_np, dtype=torch.float32, device=device)

            val_loss_tensor, val_logs = generator.loss_func(
                X_val,
                mv,
            )

            val_loss = val_loss_tensor.item()
            val_nll  = val_logs["NLL_xu"].item()

        # 스케줄러 업데이트 (val_loss 기준)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Flow ep {ep:02d}] "
            f"train_loss={train_loss:.4f}  train_NLL_xu={train_nll:.4f} | "
            f"val_loss={val_loss:.4f}  val_NLL_xu={val_nll:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # best model 저장 (val_loss 최소 기준)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(generator.state_dict())

    # 가장 좋은 val_loss 기준으로 weight 복원
    if best_state is not None:
        generator.load_state_dict(best_state)

    print(f"Best val_loss = {best_val_loss:.4f}")


# main inference
def run_feature_acquisition(
    predictor,
    generator,
    X_test,
    y_test,
    x_np,
    m_np,
    D,
    num_samples=10,
    alpha=1.0,
    gamma=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.eval()

    accs = []

    for t in range(1, D+1):
        FA = FeatureAcquisition(
            x=x_np,
            m=m_np,
            generative_model=generator,
            num_samples=num_samples,
            predictor=predictor,
            alpha=alpha,
            gamma=gamma
        )

        # mask 업데이트
        m_np, _ = FA.acquire()

        # 모델 inference
        with torch.no_grad():
            xv = X_test.to(device)
            mv = torch.tensor(m_np, dtype=torch.float32, device=device)
            logits = predictor(xv, mv)
            y_pred = logits.argmax(dim=-1).cpu().numpy()

        y_true = y_test.cpu().numpy()

        acc_t = accuracy_score(y_true, y_pred)
        accs.append(acc_t)

        print(f"Step {t}/{D} | Acc: {acc_t:.4f}")

    return accs