import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 2번 blackwell

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.append(os.path.dirname(os.getcwd()))
from src.Predictor import Predictor
from src.PartialVAE import PartialVAE
from src.Flow import Flow
from experiments.train_utils import (
    set_seed,
    train_predictor,
    train_VAE,
    train_Flow,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "Gamma-CMI", "data", "cube")

X_train = torch.load(f"{DATA_DIR}/X_train_cdf.pt").float()
y_train = torch.load(f"{DATA_DIR}/y_train.pt").long()

X_val   = torch.load(f"{DATA_DIR}/X_val_cdf.pt").float()
y_val   = torch.load(f"{DATA_DIR}/y_val.pt").long()

X_test = torch.load(f"{DATA_DIR}/X_test_cdf.pt").float()
y_test = torch.load(f"{DATA_DIR}/y_test.pt").long()


NUM_REPEATS = 5
batch_size = 128
epochs = 100
lr = 0.0003
weight_decay = 1e-4
D = 20  # feature 개수 고정

ckpt_dir = "/home/sulee/Gamma-CMI/checkpoints/cube/predictor"
os.makedirs(ckpt_dir, exist_ok=True)

for REPEAT in range(1, NUM_REPEATS+1):
    set_seed(REPEAT)
    predictor = Predictor(in_dim=20, hidden_dim=150, out_dim=8, num_hidden=2)
    optimizer = AdamW(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    train_predictor(
        predictor=predictor,
        train_loader=train_loader,
        X_val=X_val,
        y_val=y_val,
        D=D,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        metric='accuracy'
    )

    ckpt_path = os.path.join(
        ckpt_dir,
        f"predictor_cube_repeat_{REPEAT}.pt"
    )
    torch.save(predictor.state_dict(), ckpt_path)

NUM_REPEATS = 5
batch_size = 256
epochs = 100
lr = 1e-3
weight_decay = 1e-4

ckpt_dir = "/home/sulee/Gamma-CMI/checkpoints/cube/generator"
os.makedirs(ckpt_dir, exist_ok=True)


for REPEAT in range(1, NUM_REPEATS+1):
    set_seed(REPEAT)
    pvae = PartialVAE(
        input_type="continuous",
        num_con_features=20,
        num_cat_features=0,
        hidden_dim_con=150,
        most_categories=max(1, 0),   # 내부 차원 계산을 위해 최소 1
        c_dim=6,
        hid_enc=100,
        hid_dec=100,
        latent_dim=30,
        num_hidden_emb=2,
        num_hidden_enc=2,
        num_hidden_dec=2
    )
    optimizer = AdamW(pvae.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_VAE(
        generator=pvae,
        train_loader=train_loader,
        X_val=X_val,
        D=D, # feature 개수
        epochs=epochs,
        optimizer=optimizer,
        obs_sigma=0.2,
        lr_factor=0.2,
        cooldown=0,
        min_lr=1e-7,
        scheduler_patience=5
    )

    ckpt_path = os.path.join(
        ckpt_dir,
        f"VAE_cube_repeat_{REPEAT}.pt"
    )
    torch.save(pvae.state_dict(), ckpt_path)


NUM_REPEATS = 5
batch_size = 256
epochs = 100
lr = 1e-3
weight_decay = 1e-4

ckpt_dir = "/home/sulee/Gamma-CMI/checkpoints/cube/generator"
os.makedirs(ckpt_dir, exist_ok=True)

for REPEAT in range(1, NUM_REPEATS+1):
    set_seed(REPEAT)
    flow = Flow(
        num_features=D,
        hidden_dim_flow=128,
        hidden_dim_prior=128,
        num_flow_layers=4,
        num_hidden=2,
    )

    train_Flow(
        generator=flow,
        train_loader=train_loader,
        X_val=X_val,
        D=D, # feature 개수
        epochs=epochs,
        optimizer=optimizer,
        lr_factor=0.2,
        cooldown=0,
        min_lr=1e-7,
        scheduler_patience=5
    )

    ckpt_path = os.path.join(
        ckpt_dir,
        f"Flow_cube_repeat_{REPEAT}.pt"
    )
    torch.save(flow.state_dict(), ckpt_path)
