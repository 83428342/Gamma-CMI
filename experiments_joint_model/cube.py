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

from joint_model import (
    TeacherModel,
    StudentModel,
    TeacherLoss,
    StudentLoss,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # blackwell

X_train = torch.load(f"./data/cube/X_train_cdf.pt").float()
y_train = torch.load(f"./data/cube/y_train.pt").long()

X_val   = torch.load(f"./data/cube/X_val_cdf.pt").float()
y_val   = torch.load(f"./data/cube/y_val.pt").long()

X_test = torch.load(f"./data/cube/X_test_cdf.pt").float()
y_test = torch.load(f"./data/cube/y_test.pt").long()

num_features = 20
num_classes = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val,   y_val)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

z_dim          = 16   # per-feature latent dim
enc_hidden_dim = 16   # JointEncoder 내부 attention token dim
dec_hidden_dim = 64  # predictor MLP hidden dim
dec_num_hidden = 2    # predictor MLP hidden layer 개수
num_heads      = 16 # attention head의 개수

teacher = TeacherModel(
    num_features=num_features,
    z_dim=z_dim,
    enc_hidden_dim=enc_hidden_dim,
    num_heads=num_heads,
    dec_hidden_dim=dec_hidden_dim,
    dec_num_hidden=dec_num_hidden,
    out_dim=num_classes,
).to(device)

student = StudentModel(
    num_features=num_features,
    z_dim=z_dim,
    enc_hidden_dim=enc_hidden_dim,
    num_heads=num_heads,
    dec_hidden_dim=dec_hidden_dim,
    dec_num_hidden=dec_num_hidden,
    out_dim=num_classes,
).to(device)

teacher_loss_fn = TeacherLoss(
    teacher_model=teacher,
    task_type="classification",
)

student_loss_fn = StudentLoss(
    student_model=student,
    teacher_model=teacher,
    task_type="classification",
    lambda_distill=0.0,
    lambda_pred=1.0,
)

num_epochs_teacher = 100
lr_teacher = 1e-3

# optimizer 정의
optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=lr_teacher)

# 스케줄러 정의 (예: 10 epoch마다 lr 0.1배)
scheduler_teacher = StepLR(optimizer_teacher, step_size=10, gamma=0.1)

best_val_acc = 0.0
best_teacher_state = copy.deepcopy(teacher.state_dict())

# for epoch in range(1, num_epochs_teacher + 1):
#     teacher.train()
#     total_loss = 0.0
#     total_batches = 0

#     for x, y in train_loader:
#         x = x.to(device)
#         y = y.to(device)

#         # teacher는 full mask로 학습 (모든 feature 사용)
#         m_full = torch.ones_like(x, device=device)

#         out = teacher_loss_fn(x, m_full, y)
#         loss = out["loss"]

#         optimizer_teacher.zero_grad()
#         loss.backward()
#         optimizer_teacher.step()

#         total_loss += loss.item()
#         total_batches += 1

#     avg_loss = total_loss / max(total_batches, 1)

#     # validation accuracy 계산
#     val_acc = evaluate_classifier(teacher, val_loader, device)

#     # 스케줄러 스텝 (epoch 단위)
#     scheduler_teacher.step()

#     # best val 기준으로 state 저장
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         best_teacher_state = copy.deepcopy(teacher.state_dict())

#     print(
#         f"[Teacher] Epoch {epoch:02d} | loss={avg_loss:.4f} | "
#         f"val_acc={val_acc:.4f} | best_val_acc={best_val_acc:.4f}"
#     )

# # 학습 후 best validation 모델 로드
# teacher.load_state_dict(best_teacher_state)

# # teacher는 학습 끝났으니 grad 갱신 안 하도록 freeze (선택)
# for p in teacher.parameters():
#     p.requires_grad = False
# teacher.eval()

num_epochs_student = 100
lr_student = 1e-3

import copy
from torch.optim.lr_scheduler import StepLR

lr_student = 1e-3
optimizer_student = torch.optim.Adam(student.parameters(), lr=lr_student)

scheduler_student = StepLR(optimizer_student, step_size=10, gamma=0.1)

best_val_acc_student = 0.0
best_student_state = copy.deepcopy(student.state_dict())

min_K = 1
max_K = num_features

for epoch in range(1, num_epochs_student + 1):
    student.train()
    total_loss = 0.0
    total_distill = 0.0
    total_sup = 0.0
    total_batches = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        B, D = x.shape

        # student용 random mask 샘플링 (numpy -> torch)
        m_np = sample_mask_uniform_K_per_sample(
            bs=B, d=D,
            min_K=min_K, max_K=max_K
        )  # (B, D) numpy
        m_masked = torch.tensor(m_np, dtype=torch.float32, device=device)  # (B, D)

        x_masked = x * m_masked  # student가 실제로 보는 입력 값

        out = student_loss_fn(
            x_full=x,           # teacher가 보는 full feature
            x_masked=x_masked,  # student가 보는 masked feature
            m_masked=m_masked,  # student mask
            y=y,
        )
        loss = out["loss"]
        loss_distill = out["loss_distill"]
        loss_sup = out["loss_sup"]

        optimizer_student.zero_grad()
        loss.backward()
        optimizer_student.step()

        total_loss += loss.item()
        total_distill += loss_distill.item()
        total_sup += loss_sup.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_distill = total_distill / max(total_batches, 1)
    avg_sup = total_sup / max(total_batches, 1)

    # validation: full mask 기준으로 평가
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch = x_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            B, D = x_val_batch.shape

            m_np_val = sample_mask_uniform_K_per_sample(
                bs=B, d=D,
                min_K=min_K,
                max_K=max_K,
            )  # (B, D) numpy
            m_val = torch.tensor(m_np_val, dtype=torch.float32, device=device)

            x_val_masked = x_val_batch * m_val

            logits_val = student(x_val_masked, m_val)
            preds_val = logits_val.argmax(dim=-1)

            correct += (preds_val == y_val_batch).sum().item()
            total += y_val_batch.size(0)

    val_acc_student = correct / total if total > 0 else 0.0

    # 스케줄러 스텝
    scheduler_student.step()

    # best val 기준으로 학생 모델 저장
    if val_acc_student > best_val_acc_student:
        best_val_acc_student = val_acc_student
        best_student_state = copy.deepcopy(student.state_dict())

    print(
        f"[Student] Epoch {epoch:02d} | loss={avg_loss:.4f} "
        f"(distill={avg_distill:.4f}, sup={avg_sup:.4f}) | "
        f"val_acc(rand mask)={val_acc_student:.4f} | "
        f"best_val_acc={best_val_acc_student:.4f}"
    )

# 학습 종료 후 best validation 성능을 낸 student 로드
student.load_state_dict(best_student_state)
student.eval()

teacher_test_acc = evaluate_classifier(teacher, test_loader, device)

student.eval()
correct = 0
total = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        m_full_test = torch.ones_like(x_batch, device=device)
        logits = student(x_batch, m_full_test)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
student_test_acc = correct / total if total > 0 else 0.0

print("\n=== Final Test Accuracy ===")
print(f"Teacher (full mask)  test_acc = {teacher_test_acc:.4f}")
print(f"Student (full mask)  test_acc = {student_test_acc:.4f}")

save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

torch.save(best_teacher_state, os.path.join(save_dir, "teacher_cube.pt"))
torch.save(best_student_state, os.path.join(save_dir, "student_cube.pt"))

print("모델 가중치 저장 완료")

student.eval()
result = []

accs = run_feature_acquisition(
    predictor=student,
    X_test=X_test,
    y_test=y_test,
    alpha=1.0,
    gamma=0.0
)

result.append(accs)

mean_acc = np.mean(accs)
print("Mean acquisition accuracy:", mean_acc)
