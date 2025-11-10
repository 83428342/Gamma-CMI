import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim):
        super().__init__()
        self.num_classes = num_classes
        input_dim = feature_dim * 2  # feature + mask
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x, m):
        x = torch.cat([x * m, m], dim=-1)
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)

        return logits