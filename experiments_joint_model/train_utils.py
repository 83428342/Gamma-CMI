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

def run_feature_acquisition_cube(
    predictor,
    X_test,
    y_test,
    alpha=1.0,
    gamma=0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.eval()

    x = X_test.to(device)
    y_true = y_test.cpu().numpy()

    N, D = X_test.shape

    relevant_features_by_class = {
        0: np.array([0, 1, 2]),   # Class 1: features 1,2,3
        1: np.array([1, 2, 3]),   # Class 2: features 2,3,4
        2: np.array([2, 3, 4]),   # Class 3: features 3,4,5
        3: np.array([3, 4, 5]),   # Class 4: features 4,5,6
        4: np.array([4, 5, 6]),   # Class 5: features 5,6,7
        5: np.array([5, 6, 7]),   # Class 6: features 6,7,8
        6: np.array([6, 7, 8]),   # Class 7: features 7,8,9
        7: np.array([7, 8, 9]),   # Class 8: features 8,9,10
    }

    relevant_idx = np.stack(
        [relevant_features_by_class[int(c)] for c in y_true],
        axis=0
    )

    m = np.zeros((N, D), dtype=np.float32)

    accs = []
    true_feat_counts_per_step = []

    for t in range(1, D + 1):
        FA = JointFeatureAcquisition(
            x=x,
            m=m,
            predictor=predictor,
            alpha=alpha,
            gamma=gamma,
        )

        m, _ = FA.acquire()  # m: numpy (N, D)

        selected_relevant_bool = m[np.arange(N)[:, None], relevant_idx] > 0.5
        selected_relevant_count = selected_relevant_bool.sum(axis=1)

        true_feat_counts_per_step.append(selected_relevant_count)

        with torch.no_grad():
            xv = x
            mv = torch.tensor(m, dtype=torch.float32, device=device)
            logits = predictor(xv, mv)
            y_pred = logits.argmax(dim=-1).cpu().numpy()

        acc_t = accuracy_score(y_true, y_pred)
        accs.append(acc_t)

        mean_selected = selected_relevant_count.mean()
        print(
            f"Step {t}/{D} | Acc: {acc_t:.4f} | "
            f"평균 GT feature 개수: {mean_selected:.3f}"
        )

    return accs, true_feat_counts_per_step