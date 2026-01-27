import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module): # 기본 MLP 모듈
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden):
        super().__init__()
        in_dim = in_dim * 2

        if num_hidden == 0:
            self.network = nn.Linear(in_dim, out_dim)
        else:
            self.network = [ # 입력층 블럭 정의
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ]

            for _ in range(1, num_hidden): # 은닉층 블럭 정의 num_hidden 개수
                self.network.append(nn.Linear(hidden_dim, hidden_dim))
                self.network.append(nn.ReLU())
                self.network.append(nn.BatchNorm1d(hidden_dim))

            self.network.append(nn.Linear(hidden_dim, out_dim)) # 출력층 정의

            self.network = nn.Sequential(*self.network) # 블럭 조립
    
    def forward(self, x, m):
        h = torch.cat([x * m, m], dim=-1)
        return self.network(h)
