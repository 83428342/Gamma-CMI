import os
import os.path as osp
import torch

from Utils import Utils

class Dataset:
    def __init__(self, seed):
        Utils.set_seed(seed)
        self.seed = seed

    def generate_cube(
        self,
        path,
        train_size=60_000,
        val_size=10_000,
        test_size=10_000,
        noise_mu=0.5,
        noise_sig=0.3,
        num_features=20,
        num_classes=8,
    ):
        os.makedirs(path, exist_ok=True)

        dataset_dict = {
            "dataset": "cube",
            "num_con_features": num_features,
            "num_cat_features": 0,
            "most_categories": 0,
            "out_dim": num_classes,
            "metric": "accuracy",
            "max_dim": None,
        }

        means = torch.full((num_classes, num_features), noise_mu)
        means[0, 0:3] = torch.tensor([0.0, 0.0, 0.0])
        means[1, 1:4] = torch.tensor([1.0, 0.0, 0.0])
        means[2, 2:5] = torch.tensor([0.0, 1.0, 0.0])
        means[3, 3:6] = torch.tensor([1.0, 1.0, 0.0])
        means[4, 4:7] = torch.tensor([0.0, 0.0, 1.0])
        means[5, 5:8] = torch.tensor([1.0, 0.0, 1.0])
        means[6, 6:9] = torch.tensor([0.0, 1.0, 1.0])
        means[7, 7:10] = torch.tensor([1.0, 1.0, 1.0])

        sigs = torch.full((num_classes, num_features), noise_sig)
        sigs[0, 0:3] = torch.tensor([0.1, 0.1, 0.1])
        sigs[1, 1:4] = torch.tensor([0.1, 0.1, 0.1])
        sigs[2, 2:5] = torch.tensor([0.1, 0.1, 0.1])
        sigs[3, 3:6] = torch.tensor([0.1, 0.1, 0.1])
        sigs[4, 4:7] = torch.tensor([0.1, 0.1, 0.1])
        sigs[5, 5:8] = torch.tensor([0.1, 0.1, 0.1])
        sigs[6, 6:9] = torch.tensor([0.1, 0.1, 0.1])
        sigs[7, 7:10] = torch.tensor([0.1, 0.1, 0.1])

        size_per_class = (train_size + val_size + test_size) // num_classes
        X, y = self._create_data(size_per_class, means, sigs, num_features, num_classes)

        assert X.shape[0] == train_size + val_size + test_size, "size mismatch"
        assert torch.all(y[0:8] == torch.arange(num_classes)), "label order mismatch"

        self.preprocess_and_save_data_minimal(
            path=path,
            dataset_dict=dataset_dict,
            train_size=train_size,
            val_size=val_size,
            X=X,
            y=y,
            M=None,
            shuffle=False,
        )

    def _create_data(self, size_per_class, means, sigs, num_features, num_classes):
        X_data = torch.distributions.normal.Normal(means, sigs).sample([size_per_class])
        X_data = X_data.view(-1, num_features).float()
        y_data = torch.arange(num_classes).repeat(size_per_class).long()
        return X_data, y_data

    @torch.no_grad()
    def preprocess_and_save_data_minimal(
        self,
        path: str,
        dataset_dict: dict,
        train_size: int,
        val_size: int,
        X: torch.Tensor,
        y: torch.Tensor,
        M: torch.Tensor | None = None,
        shuffle: bool = False,
    ):
        os.makedirs(path, exist_ok=True)
        assert X.shape[0] == y.shape[0], "X/y length mismatch"
        total = X.shape[0]
        assert train_size + val_size < total, "split sizes too large"
        if M is not None:
            assert M.shape == X.shape, "M shape mismatch"

        idx = torch.randperm(total) if shuffle else torch.arange(total)
        tr_idx = idx[:train_size]
        va_idx = idx[train_size:train_size + val_size]
        te_idx = idx[train_size + val_size:]

        X_train, X_val, X_test = X[tr_idx], X[va_idx], X[te_idx]
        y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]
        if M is not None:
            M_train, M_val, M_test = M[tr_idx], M[va_idx], M[te_idx]
        else:
            M_train = M_val = M_test = None

        torch.save(dataset_dict, osp.join(path, "dataset_dict.pt"))
        torch.save(X_train, osp.join(path, "X_train.pt"))
        torch.save(X_val,   osp.join(path, "X_val.pt"))
        torch.save(X_test,  osp.join(path, "X_test.pt"))
        torch.save(y_train.long(), osp.join(path, "y_train.pt"))
        torch.save(y_val.long(),   osp.join(path, "y_val.pt"))
        torch.save(y_test.long(),  osp.join(path, "y_test.pt"))
        if M_train is not None:
            torch.save(M_train, osp.join(path, "M_train.pt"))
            torch.save(M_val,   osp.join(path, "M_val.pt"))
            torch.save(M_test,  osp.join(path, "M_test.pt"))

if __name__ == "__main__":
    Utils.set_seed(8917)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "data", "Cube")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not (os.path.exists(f"{DATA_DIR}/X_train.pt") and os.path.exists(f"{DATA_DIR}/y_train.pt")):
        ds = Dataset(seed=42)
        ds.generate_cube(path=DATA_DIR)

    dataset_dict = torch.load(f"{DATA_DIR}/dataset_dict.pt")
    X_train = torch.load(f"{DATA_DIR}/X_train.pt").numpy()
    X_val   = torch.load(f"{DATA_DIR}/X_val.pt").numpy()
    y_train = torch.load(f"{DATA_DIR}/y_train.pt").numpy()
    y_val   = torch.load(f"{DATA_DIR}/y_val.pt").numpy()
    X_test  = torch.load(f"{DATA_DIR}/X_test.pt").numpy()
    y_test  = torch.load(f"{DATA_DIR}/y_test.pt").numpy()

    feature_dim = X_train.shape[1]
    num_classes = int(dataset_dict["out_dim"])