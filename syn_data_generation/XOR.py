import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def create_XOR(n, noisy_x, delta, seed=42):
    """
    XOR synthetic dataset 생성 함수.
    X = [x1, x2, noisy_features]
    y = x1 XOR x2
    delta는 noisy feature가 y와 align되는 정도 (0.0 ~ 0.5)
    """
    assert 0.0 <= delta <= 0.5

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Base XOR data 생성
    x1 = np.random.binomial(1, 0.5, size=n)
    x2 = np.random.binomial(1, 0.5, size=n)
    y = (x1 ^ x2).astype(int)

    # noisy feature 확률 조정
    p = 0.5 + delta * (2 * y - 1)
    p = p.reshape(-1, 1)

    noisy = np.random.binomial(1, p, size=(n, noisy_x))

    # Final X = [x1, x2, noisy_x...]
    X = np.concatenate([
        x1.reshape(-1, 1),
        x2.reshape(-1, 1),
        noisy
    ], axis=-1)

    # Train / Val / Test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.20, random_state=seed, shuffle=True
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, shuffle=True
    )

    # torch tensors
    X_train = torch.from_numpy(X_train).long()
    X_val   = torch.from_numpy(X_val).long()
    X_test  = torch.from_numpy(X_test).long()

    y_train = torch.from_numpy(y_train).long()
    y_val   = torch.from_numpy(y_val).long()
    y_test  = torch.from_numpy(y_test).long()

    # 저장 경로 설정
    #   /Gamma-CMI_ver1/data/XOR_problem_delta{delta}/
    file_dir = os.path.dirname(os.path.abspath(__file__))     # syn_data_generation/
    root_dir = os.path.dirname(file_dir)                      # 프로젝트 루트

    save_dir = os.path.join(root_dir, "data", f"XOR_problem_delta{delta}")
    os.makedirs(save_dir, exist_ok=True)

    # 저장
    torch.save(X_train, os.path.join(save_dir, "X_train.pt"))
    torch.save(X_val,   os.path.join(save_dir, "X_val.pt"))
    torch.save(X_test,  os.path.join(save_dir, "X_test.pt"))

    torch.save(y_train, os.path.join(save_dir, "y_train.pt"))
    torch.save(y_val,   os.path.join(save_dir, "y_val.pt"))
    torch.save(y_test,  os.path.join(save_dir, "y_test.pt"))

    print(f"[✔] XOR dataset saved to {save_dir}")


if __name__ == "__main__":
    # 기본 설정
    n = 10000         # 전체 데이터 수
    noisy_x = 4       # noisy feature 개수
    delta = 0.0       # 0.0 ~ 0.5

    print(f"Generating XOR synthetic data: n={n}, noisy_x={noisy_x}, delta={delta}")
    create_XOR(n=n, noisy_x=noisy_x, delta=delta)
