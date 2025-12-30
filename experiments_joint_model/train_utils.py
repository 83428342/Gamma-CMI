import sys, os
import numpy as np

import torch
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from joint_model import JointFeatureAcquisition

def sample_mask_uniform_K_per_sample(bs, d, min_K, max_K): # batch size, feature 개수, 최소 관측 샘플 수, 최대 관측 샘플 수
    m = np.zeros((bs, d), dtype=np.float32)
    Ks = np.random.randint(min_K, max_K+1, size=(bs,))
    for i, K in enumerate(Ks): # Ks의 index와 해당 index의 값
        idx = np.random.choice(d, size=K, replace=False)
        m[i, idx] = 1.0
    return m

def evaluate_classifier(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            # Teacher는 full mask (모든 feature 관측) 가정
            m_full = torch.ones_like(x, device=device)
            logits = model(x, m_full)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total > 0 else 0.0
    return acc

def run_feature_acquisition(
    predictor,
    X_test,
    y_test,
    alpha=1.0,
    gamma=0.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.eval()

    x = X_test.to(device)

    N, D = X_test.shape
    m = np.zeros((N, D), dtype=np.float32)

    accs = []

    # y_true는 numpy로 미리 빼두자
    y_true = y_test.cpu().numpy()

    for t in range(1, D + 1):
        FA = JointFeatureAcquisition(
            x=x,
            m=m,
            predictor=predictor,
            alpha=alpha,
            gamma=gamma,
        )

        m, _ = FA.acquire()

        with torch.no_grad():
            xv = x
            mv = torch.tensor(m, dtype=torch.float32, device=device)
            logits = predictor(xv, mv)
            y_pred = logits.argmax(dim=-1).cpu().numpy()

        acc_t = accuracy_score(y_true, y_pred)
        accs.append(acc_t)

        print(f"Step {t}/{D} | Acc: {acc_t:.4f}")

    return accs
