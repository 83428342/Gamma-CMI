import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MLP

class JointEncoder(nn.Module):
    def __init__(
        self,
        num_features,
        z_dim, # 각 feature의 출력층 dim
        enc_hidden_dim, # hidden layer의 dim
        num_heads # attention head의 개수
    ):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads

        self.input_proj = nn.Linear(2, enc_hidden_dim) # token embedding [x * m, m]

        self.attention = nn.MultiheadAttention(
            embed_dim=enc_hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(enc_hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(enc_hidden_dim, enc_hidden_dim * 4),
            nn.GELU(),
            nn.Linear(enc_hidden_dim * 4, enc_hidden_dim),
        )
        self.ln2 = nn.LayerNorm(enc_hidden_dim)

        self.out_proj = nn.Linear(enc_hidden_dim, z_dim)

    def forward(self, x, m):
        x_masked = x * m
        h = torch.stack([x_masked, m], dim=-1)
        token = self.input_proj(h) # token embedding

        batch_size, seq_len, _ = token.shape
        eye = torch.eye(seq_len, device=token.device).unsqueeze(0)
        m_rows = m.unsqueeze(-1)

        base = (1.0 - eye) * m_rows
        inf_value = float('-inf')
        attn_mask = torch.zeros_like(base)
        attn_mask[base > 0.5] = inf_value # 부동소수점에 의한 오류 방지
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        attn_out, attn_weights = self.attention(
            token, # query
            token, # key
            token, # value
            attn_mask=attn_mask # attention mask
        )
        
        h1 = self.ln1(token + attn_out)
        ffn_out = self.ffn(h1)
        h2 = self.ln2(h1 + ffn_out)
        z_raw = self.out_proj(h2)
        return z_raw
    
class JointDecoder(nn.Module):
    def __init__(
        self,
        num_features,
        z_dim,
        dec_hidden_dim,
        dec_num_hidden,
        num_heads_dec,
        out_dim
    ):
        super().__init__()
        self.num_features = num_features
        self.z_dim = z_dim

        # 디코더의 transformer block
        self.attention = nn.MultiheadAttention(
            embed_dim=z_dim,
            num_heads=num_heads_dec,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(z_dim)

        self.ffn = nn.Sequential(
            nn.Linear(z_dim, z_dim * 4),
            nn.GELU(),
            nn.Linear(z_dim * 4, z_dim),
        )
        self.ln2 = nn.LayerNorm(z_dim)

        # 디코더의 최종 MLP block
        self.mlp = MLP(
            in_dim=num_features * z_dim,
            hidden_dim=dec_hidden_dim,
            out_dim=out_dim,
            num_hidden=dec_num_hidden
        )

    def forward(self, z):
        B, D, Z = z.shape
        # Transformer
        attn_out, attn_weights = self.attention(
            z, # query
            z, # key
            z, # value
        )

        h1 = self.ln1(z + attn_out)
        ffn_out = self.ffn(h1)
        h2 = self.ln2(h1 + ffn_out)

        h_flat = h2.view(B, D * Z)
        # MLP
        logit = self.mlp(h_flat)
        return logit

    
class JointModel(nn.Module):
    def __init__(
        self,
        num_features,
        z_dim,
        enc_hidden_dim,
        num_heads, # attention head의 개수
        dec_hidden_dim, # 디코더 hidden layer의 dim
        dec_num_hidden, # 디코더 hidden layer 개수
        out_dim # 클래수 개수 (regression에서는 1)
    ):
        super().__init__()

        self.encoder = JointEncoder(
            num_features=num_features,
            z_dim=z_dim,
            enc_hidden_dim=enc_hidden_dim,
            num_heads=num_heads, # attention head의 개수
        )

        self.dec_predictor = JointDecoder(
            num_features=num_features,
            z_dim=z_dim,
            dec_hidden_dim=dec_hidden_dim,
            dec_num_hidden=dec_num_hidden,
            num_heads_dec=num_heads,
            out_dim=out_dim
        )

        self.dec_acquisition = MLP(
            in_dim=num_features * z_dim,
            hidden_dim=dec_hidden_dim,
            out_dim=out_dim,
            num_hidden=dec_num_hidden
        )

    def forward(self, x, m):
        z_raw = self.encoder(x, m)
        z = z_raw * m.unsqueeze(-1)
        logits = self.dec_predictor(z)
        return logits
    
    def forward_acq_logits(self, x, m):
        z_raw = self.encoder(x, m)
        z = z_raw * m.unsqueeze(-1)

        B, D, Z = z.shape
        z = z.view(B, D * Z)

        logits = self.dec_acquisition(z)
        return logits
    

class Stage1Loss(nn.Module):
    def __init__(self, model, task_type):
        super().__init__()
        self.model = model
        self.task_type = task_type

        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.criterion = nn.MSELoss()

    def forward(self, x_masked, m_masked, y):
        logit = self.model.forward_acq_logits(x_masked, m_masked)

        if self.task_type == "classification":
            loss = self.criterion(logit, y)
        else:
            logit_flat = logit.view(-1)
            y_flat = y.view(-1).float()
            loss = self.criterion(logit_flat, y_flat)

        return {"loss": loss}


class Stage2Loss(nn.Module):
    def __init__(self, model, task_type):
        super().__init__()
        self.model = model
        self.task_type = task_type

        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task_type == "regression":
            self.criterion = nn.MSELoss()

    def forward(self, x_masked, m_masked, y):
        logit = self.model(x_masked, m_masked)

        if self.task_type == "classification":
            loss = self.criterion(logit, y)
        else:
            logit_flat = logit.view(-1)
            y_flat = y.view(-1).float()
            loss = self.criterion(logit_flat, y_flat)

        return {"loss": loss}


class JointFeatureAcquisition():
    def __init__(self, x, m, model, alpha=1, gamma=0):
        self.x = x
        self.m = m
        self.model = model
        self.alpha = alpha
        self.gamma = gamma

    def entropy(self, p,  eps=1e-10):
        alpha = self.alpha
        p = np.clip(p, eps, 1.0)  

        if alpha == 0.0:
            return np.log((p > eps).sum(axis=-1) + eps)
        
        elif alpha == 1.0:
            return -np.sum(p * np.log(p + eps), axis=-1)

        elif alpha > 1000:
            return -np.log(np.max(p, axis=-1) + eps)
        
        else:
            return (1.0 / (1.0 - alpha)) * np.log(np.sum(np.power(p, alpha), axis=-1) + eps)

    def alpha_gamma_cmi(self):
        x = self.x
        m = self.m
        gamma = self.gamma
        model = self.model

        device = next(model.parameters()).device

        m_upsampled = np.random.binomial(n=1, p=gamma, size=m.shape) # 각 feature별로 0 또는 1로 변형 
        m_repeated = np.maximum(m, m_upsampled) # 위에서는 모든 feature별로 진행했으니 max로 병합
        
        m_repeated = torch.tensor(m_repeated, dtype=torch.float32, device=device)

        with torch.no_grad():
            m = torch.tensor(m, dtype=torch.float32, device=device)
            z_base = model.encoder(x, m) # 기본 z 얻어놓기
        B, D, Z = z_base.shape

        out = []

        # 변수 하나씩 mask해가며 entropy 계산
        for f in range(D):
            # without 계산
            m_without = m_repeated.clone()
            m_without[:, f] = 0.0
            z_without = z_base * m_without.unsqueeze(-1)

            with torch.no_grad():
                z_without = z_without.view(B, D * Z)
                logits_without = model.dec_acquisition(z_without)
                p_without = torch.softmax(logits_without, dim=-1).cpu().numpy()
            h_without = self.entropy(p=p_without)

            # with 계산
            m_with = m_repeated.clone()
            m_with[:, f] = 1.0
            z_with = z_base * m_with.unsqueeze(-1)

            with torch.no_grad():
                z_with = z_with.view(B, D * Z)
                logits_with = model.dec_acquisition(z_with)
                p_with = torch.softmax(logits_with, dim=-1).cpu().numpy()
            h_with = self.entropy(p=p_with)

            # 차이 계산
            entropy_diff = h_without - h_with
            out.append(entropy_diff)
            
        return np.stack(out, axis=-1)

    def acquire(self):
        m = self.m
        scores = self.alpha_gamma_cmi()
        scores -= scores.min()
        scores *= (1 - m)
        scores += 1e-10 * (1 - m) * np.random.uniform(size=(scores.shape)) # 최고점 score 점수 같음 방지

        selected = np.argmax(scores, axis=-1)
        m[np.arange(m.shape[0]), selected] = 1.0
        self.m = m # acquire 후 해당 feature의 mask = 1로 변경 

        return m, selected