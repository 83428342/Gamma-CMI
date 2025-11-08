import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim):
        super().__init__()
        self.num_classes = num_classes
        input_dim = feature_dim * 2  # feature + mask
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, m):
        x = torch.cat([x * m, m], dim=-1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)

        return F.softmax(logits, dim=-1) # 다중 분류용이라 이진 분류 해야하면 다시 만들어야함 