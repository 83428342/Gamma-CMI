import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def generate_indicator_dataset(
    d=6,
    n=10000,
    seed=42,
    save_dir=None,
):
    """
    d: feature 개수 (양자택일 인덱스 후보 수)
    n: 총 샘플 수
    save_dir: 저장할 폴더 경로
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 데이터 생성
    X = np.random.binomial(n=1, p=0.5, size=(n, d))

    indicators = np.random.choice(a=d, size=(n, 1), replace=True)
    y = X[np.arange(n), indicators.flatten()]  # label = 선택된 인덱스 feature 값

    # indicator feature 포함
    X = np.concatenate([X, indicators], axis=-1)

    # Train / Val / Test 분할
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    # Torch tensor 변환
    X_train = torch.from_numpy(X_train).float()
    X_val   = torch.from_numpy(X_val).float()
    X_test  = torch.from_numpy(X_test).float()

    y_train = torch.from_numpy(y_train).float()
    y_val   = torch.from_numpy(y_val).float()
    y_test  = torch.from_numpy(y_test).float()

    # 5. 저장 경로 설정
    if save_dir is None:
        # indicator.py의 상위 디렉토리를 프로젝트 루트로 간주
        FILE_DIR = os.path.dirname(os.path.abspath(__file__))        # syn_data_generation/
        ROOT_DIR = os.path.dirname(FILE_DIR)                         # Gamma-CMI_ver1/
        save_dir = os.path.join(ROOT_DIR, "data", "indicator")       # data/indicator/

    os.makedirs(save_dir, exist_ok=True)

    # 저장
    torch.save(X_train, os.path.join(save_dir, "X_train.pt"))
    torch.save(X_val,   os.path.join(save_dir, "X_val.pt"))
    torch.save(X_test,  os.path.join(save_dir, "X_test.pt"))

    torch.save(y_train, os.path.join(save_dir, "y_train.pt"))
    torch.save(y_val,   os.path.join(save_dir, "y_val.pt"))
    torch.save(y_test,  os.path.join(save_dir, "y_test.pt"))

    print(f"Saved dataset to: {save_dir}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


if __name__ == "__main__":
    generate_indicator_dataset()
