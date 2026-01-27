import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.layers import MLP, StochasticMLP


# 기본 RealNVP 블럭 
class RealNVPLayer(nn.Module):
    def __init__(self, num_features, num_conditioning_features, hidden_dim, num_hidden):
        super().__init__()
        self.za_dim = int(num_features / 2)
        self.zb_dim = num_features - self.za_dim
        in_dim = self.zb_dim + num_conditioning_features

        # scale network
        self.scale_network = nn.Sequential(
            MLP(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=self.za_dim,
                num_hidden=num_hidden,
            ),
        nn.Tanh(),
        )

        # shift network
        self.shift_network = nn.Sequential(
            MLP(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=self.za_dim,
                num_hidden=num_hidden,
            ),
        nn.Tanh(),
        )

        perm = torch.randperm(num_features)
        self.register_buffer("perm", perm) # 변형
        self.register_buffer("inv_perm", torch.argsort(perm)) # 복원 

    def split(self, z):
        z = z[:, self.perm]
        return z[:, :self.za_dim], z[:, self.za_dim:]

    def recombine(self, za, zb):
        return torch.cat([za, zb], dim=-1)[:, self.inv_perm]

    def forward(self, z, conditioning):
        za, zb = self.split(z)
        input = torch.cat([zb, conditioning], dim=-1)
        s = self.scale_network(input)
        t = self.shift_network(input)
        za = za * torch.exp(s) + t
        z = self.recombine(za, zb)
        log_det = torch.sum(s, dim=-1)
        return z, log_det

    def inverse(self, z, conditioning):
        za, zb = self.split(z)
        input = torch.cat([zb, conditioning], dim=-1)
        s = self.scale_network(input)
        t = self.shift_network(input)
        za = (za - t) * torch.exp(-s)
        z = self.recombine(za, zb)
        log_det = torch.sum(-s, dim=-1)
        return z, log_det


# 전체 RealNVP 구조
class FlowNetwork(nn.Module):
    def __init__(self, num_features, num_conditioning_features, hidden_dim, num_flow_layers, num_hidden):
        super().__init__()
        self.num_features = num_features
        self.num_conditioning_features = num_conditioning_features

        self.flow_layers = nn.ModuleList([
            RealNVPLayer(
                num_features=num_features,
                num_conditioning_features=num_conditioning_features,
                hidden_dim=hidden_dim,
                num_hidden=num_hidden
            )
            for _ in range(num_flow_layers)
        ])

    def forward(self, z, conditioning): # 학습 때 사용
        log_det_total = torch.zeros(z.size(0), device=z.device)
        for layer in self.flow_layers:
            z, log_det = layer.forward(z, conditioning)
            log_det_total = log_det_total + log_det
        return z, log_det_total
    
    def inverse(self, z, conditioning): # 생성 때 사용
        log_det_total = torch.zeros(z.size(0), device=z.device)
        # 역변환은 layer 순서를 반대로
        for layer in reversed(self.flow_layers):
            z, log_det = layer.inverse(z, conditioning)
            log_det_total = log_det_total + log_det
        return z, log_det_total


# latent sampling
class PriorNetwork(nn.Module):
    def __init__(self,num_features, num_conditioning_features, hidden_dim, num_hidden):
        super().__init__()
        self.out_dim = num_features

        self.prior_network = StochasticMLP(
            in_dim=num_conditioning_features,
            hidden_dim=hidden_dim,
            out_dim=num_features,
            num_hidden=num_hidden,
        )

    def get_prior(self, conditioning):
        mean, sig = self.prior_network(conditioning)
        return Normal(loc=mean, scale=sig)
    
    def log_likelihood(self, z, conditioning):
        prior = self.get_prior(conditioning)
        return prior.log_prob(z).sum(dim=-1)
    
    def conditional_latent_sample(self, conditioning, num_samples):
        prior = self.get_prior(conditioning)

        samples = prior.sample((num_samples,))
        samples = samples.transpose(0, 1)
        samples = samples.reshape(-1, self.out_dim)
        return samples
    

class Flow(nn.Module):
    def __init__(self, num_features, hidden_dim_flow, hidden_dim_prior, num_flow_layers, num_hidden):
        super().__init__()
        self.num_features = num_features

        # conditioning: [x_o, m] → 2 * D
        num_conditioning_features = 2 * num_features

        self.flow = FlowNetwork(
            num_features=num_features,
            num_conditioning_features=num_conditioning_features,
            hidden_dim=hidden_dim_flow,
            num_flow_layers=num_flow_layers,
            num_hidden=num_hidden
        )

        self.prior = PriorNetwork(
            num_features=num_features,
            num_conditioning_features=num_conditioning_features,
            hidden_dim=hidden_dim_prior,
            num_hidden=num_hidden
        )

    def get_xu_conditioning(self, x, m):
        x_o = x * m            # observed
        x_u = x * (1.0 - m)    # unobserved (학습 시 기준값)

        # conditioning 벡터: [x_o, m]
        conditioning = torch.cat([x_o, m], dim=-1)

        return x_u, conditioning
    
    def nll(self, x, m):
        # x_u, cond 만들기
        x_u, conditioning = self.get_xu_conditioning(x, m)

        # flow: x_u -> z
        z, log_det = self.flow.forward(x_u, conditioning)

        # prior log p(z | cond)
        prior_log_likelihood = self.prior.log_likelihood(z, conditioning)

        # log p(x_u | x_o, m) = log p(z|cond) + log|det dz/dx_u|
        log_px_u = prior_log_likelihood + log_det
        nll = -log_px_u.mean() / float(self.num_features)

        return nll

    def loss_func(self, x, m):
        nll = self.nll(x, m)
        logs = {"NLL_xu": nll.detach()}
        return nll, logs
    
    def conditional_samples(self, x, m, num_samples):
        _, conditioning = self.get_xu_conditioning(x, m)

        # latent z_u ~ p(z | cond)
        z_u = self.prior.conditional_latent_sample(conditioning, num_samples)
        conditioning_rep = conditioning.repeat_interleave(num_samples, dim=0)

        # z_u → x_u
        x_u, _ = self.flow.inverse(z_u, conditioning_rep)

        # 관측된 부분은 x 유지, 나머지 채우기
        x_rep = x.repeat_interleave(num_samples, dim=0)
        m_rep = m.repeat_interleave(num_samples, dim=0)

        x_filled = m_rep * x_rep + (1.0 - m_rep) * x_u
        return x_filled

    @torch.no_grad()
    def generate(self, x, m, num_samples: int = 1):
        self.eval()
        device = next(self.parameters()).device

        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        mt = torch.as_tensor(m, dtype=torch.float32, device=device)

        x_filled = self.conditional_samples(
            xt,
            m=mt,
            num_samples=num_samples
        )

        return x_filled.detach().cpu().numpy()