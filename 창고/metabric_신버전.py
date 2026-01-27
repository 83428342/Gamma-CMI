import os
import copy
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from train_utils import (
    evaluate_classifier,
    sample_mask_uniform_K_per_sample,
    run_feature_acquisition
)

# (변경) teacher/student 대신 JointModel + Stage1Loss/Stage2Loss import
from joint_model_신버전 import (
    JointModel,
    Stage1Loss,
    Stage2Loss,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # blackwell

X_train = torch.load(f"./data/metabric/X_train_cdf.pt").float()
y_train = torch.load(f"./data/metabric/y_train.pt").long()

X_val   = torch.load(f"./data/metabric/X_val_cdf.pt").float()
y_val   = torch.load(f"./data/metabric/y_val.pt").long()

X_test  = torch.load(f"./data/metabric/X_test_cdf.pt").float()
y_test  = torch.load(f"./data/metabric/y_test.pt").long()

num_features = 12
num_classes  = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val,   y_val)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

z_dim          = 64    # per-feature latent dim
enc_hidden_dim = 8     # JointEncoder 내부 attention token dim
dec_hidden_dim = 256   # decoder MLP hidden dim
dec_num_hidden = 2     # decoder MLP hidden layer 개수
num_heads      = 8     # attention head의 개수

model = JointModel(
    num_features=num_features,
    z_dim=z_dim,
    enc_hidden_dim=enc_hidden_dim,
    num_heads=num_heads,
    dec_hidden_dim=dec_hidden_dim,
    dec_num_hidden=dec_num_hidden,
    out_dim=num_classes,
).to(device)

stage1_loss_fn = Stage1Loss(
    model=model,
    task_type="classification",
)

num_epochs_stage1 = 100
lr_stage1 = 1e-3

optimizer_stage1 = torch.optim.Adam(
    list(model.encoder.parameters()) + list(model.dec_acquisition.parameters()),
    lr=lr_stage1
)
scheduler_stage1 = StepLR(optimizer_stage1, step_size=10, gamma=0.1)

best_val_acc_stage1 = 0.0
best_stage1_state = copy.deepcopy(model.state_dict())

min_K = 1
max_K = num_features

for epoch in range(1, num_epochs_stage1 + 1):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        B, D = x.shape

        m_np = sample_mask_uniform_K_per_sample(
            bs=B, d=D,
            min_K=min_K, max_K=max_K
        )
        m_masked = torch.tensor(m_np, dtype=torch.float32, device=device)
        x_masked = x * m_masked

        out = stage1_loss_fn(x_masked, m_masked, y)
        loss = out["loss"]

        optimizer_stage1.zero_grad()
        loss.backward()
        optimizer_stage1.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch = x_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)

            m_full_val = torch.ones_like(x_val_batch, device=device)
            logits_val = model.forward_acq_logits(x_val_batch, m_full_val)
            preds_val = logits_val.argmax(dim=-1)
            correct += (preds_val == y_val_batch).sum().item()
            total += y_val_batch.size(0)
    val_acc_stage1 = correct / total if total > 0 else 0.0

    scheduler_stage1.step()

    if val_acc_stage1 > best_val_acc_stage1:
        best_val_acc_stage1 = val_acc_stage1
        best_stage1_state = copy.deepcopy(model.state_dict())

    print(
        f"[Stage1: enc+acqMLP] Epoch {epoch:02d} | loss={avg_loss:.4f} | "
        f"val_acc(acq, full mask)={val_acc_stage1:.4f} | best_val_acc={best_val_acc_stage1:.4f}"
    )

model.load_state_dict(best_stage1_state)

for p in model.encoder.parameters():
    p.requires_grad = False
for p in model.dec_acquisition.parameters():
    p.requires_grad = False

stage2_loss_fn = Stage2Loss(
    model=model,
    task_type="classification",
)

num_epochs_stage2 = 100
lr_stage2 = 1e-3

optimizer_stage2 = torch.optim.Adam(
    list(model.dec_predictor.parameters()),
    lr=lr_stage2
)
scheduler_stage2 = StepLR(optimizer_stage2, step_size=10, gamma=0.1)

best_val_acc_stage2 = 0.0
best_stage2_state = copy.deepcopy(model.state_dict())

for epoch in range(1, num_epochs_stage2 + 1):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        B, D = x.shape

        m_np = sample_mask_uniform_K_per_sample(
            bs=B, d=D,
            min_K=min_K, max_K=max_K
        )
        m_masked = torch.tensor(m_np, dtype=torch.float32, device=device)
        x_masked = x * m_masked

        out = stage2_loss_fn(x_masked, m_masked, y)
        loss = out["loss"]

        optimizer_stage2.zero_grad()
        loss.backward()
        optimizer_stage2.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch = x_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)

            m_full_val = torch.ones_like(x_val_batch, device=device)
            logits_val = model(x_val_batch, m_full_val)
            preds_val = logits_val.argmax(dim=-1)
            correct += (preds_val == y_val_batch).sum().item()
            total += y_val_batch.size(0)
    val_acc_stage2 = correct / total if total > 0 else 0.0

    scheduler_stage2.step()

    if val_acc_stage2 > best_val_acc_stage2:
        best_val_acc_stage2 = val_acc_stage2
        best_stage2_state = copy.deepcopy(model.state_dict())

    print(
        f"[Stage2: predDecoder] Epoch {epoch:02d} | loss={avg_loss:.4f} | "
        f"val_acc(pred, full mask)={val_acc_stage2:.4f} | best_val_acc={best_val_acc_stage2:.4f}"
    )

# (변경) stage2 best 로드 (최종 모델)
model.load_state_dict(best_stage2_state)
model.eval()

model_test_acc = evaluate_classifier(model, test_loader, device)

print("\n=== Final Test Accuracy ===")
print(f"JointModel (pred, full mask)  test_acc = {model_test_acc:.4f}")

save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

torch.save(best_stage1_state, os.path.join(save_dir, "joint_stage1_metabric.pt"))
torch.save(best_stage2_state, os.path.join(save_dir, "joint_stage2_metabric.pt"))

print("모델 가중치 저장 완료")

model.eval()
result = []

accs = run_feature_acquisition(
    model=model,
    X_test=X_test,
    y_test=y_test,
    alpha=1.0,
    gamma=0.0
)

result.append(accs)

mean_acc = np.mean(accs)
print("Mean acquisition accuracy:", mean_acc)
print(result)
