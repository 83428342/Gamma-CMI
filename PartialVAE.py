import torch
import torch.nn as nn
import torch.nn.functional as F

'''
전체 구조
- x -> c로 보냄(각 feature를 embedding, unobserved는 0으로)
- c -> z로 보냄(gaussian의 mean, std 뽑은 후 sampling)
- z -> h로 보냄
- h -> x를 생성
'''

# 전체 loss = NLL_con + NLL_cat + KL 

def kl_01_loss(mu, sig):
    # KL[q(z|x)=N(mu, sig^2) || N(0, I)]
    # 배치 평균으로 반환
    kl = 0.5 * (mu.pow(2) + sig.pow(2) - (sig.log() * 2) - 1.0)
    return kl.sum(dim=1).mean()

def nll_continuous(x_true, x_pred_mean, mask, obs_sigma=1.0):
    # mask==1인 위치만 집계
    var = obs_sigma ** 2
    sq = (x_true - x_pred_mean).pow(2)
    nll = 0.5 * (sq / var)
    # 관측치만 평균
    denom = mask.sum().clamp_min(1.0)
    return (nll * mask).sum() / denom

def nll_categorical(x_true_idx, logits, mask):
    B, Fcat = x_true_idx.shape
    C = logits.shape[1] // Fcat
    logits = logits.view(B, Fcat, C)
    # CE 계산 (feature 차원마다)
    ce = F.cross_entropy(
        logits.permute(0,2,1),
        x_true_idx,
        reduction='none'
    )
    denom = mask.sum().clamp_min(1.0)
    return (ce * mask).sum() / denom

# 연속형 변수의 임베딩 클래스
class ContinuousXtoC(nn.Module):
    def __init__(self, num_con_features, hidden_dim, c_dim): # 연속형 변수의 개수, MLP 중간층 뉴런 개수, 최종 임베딩 차원
        super().__init__()
        self.E = nn.Parameter(torch.randn(1, num_con_features, c_dim)) # 각 feature마다 다르게 임베딩해서 표현
        self.fc1 = nn.Linear(c_dim+1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, c_dim)

    def forward(self, x, m):
        x = x.unsqueeze(-1)
        x_embedded = x * self.E # 각 feature를 임베딩한 행렬
        s = torch.cat([x, x_embedded], dim=-1) # 원래 x와 concat해서 표현

        h = F.relu(self.fc1(s)) # 행별로 MLP 연산
        u = self.fc2(h) 

        c = (u * m.unsqueeze(-1)).sum(dim=1) # mask가 1인 모든 c를 병합
        return c 

# 범주형 변수의 임베딩 클래스
class CategoricalXtoC(nn.Module):
    def __init__(self, num_cat_features, most_categories, c_dim): # 범주형 변수의 개수, 범주가 가장 많은 변수의 범주 개수, 최종 임베딩 차원
        super().__init__()
        self.E = nn.Embedding(num_cat_features*most_categories, c_dim)
        self.shift = (torch.arange(num_cat_features)*(most_categories)).unsqueeze(0).long()
        self.shift = nn.Parameter(data=self.shift, requires_grad=False)

    def forward(self, x, m):
        x = x.long() + self.shift
        x = self.E(x)

        c = torch.sum(x*m.unsqueeze(-1), dim=1)
        return c        

# c를 입력으로 받아 gaussian을 출력하는 클래스
class CtoZ(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*out_dim)
        self.out_dim = out_dim

    def forward(self, c):
        c = F.relu(self.fc1(c))
        c = self.fc2(c)

        mu = c[:, :self.out_dim]
        sig = c[:, self.out_dim:]

        sig = F.softplus(sig)
        sig = sig + 1e-4

        return mu, sig

class ZtoH(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn  = nn.BatchNorm1d(hidden_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = self.bn(h)
        return h
    
# 주의사항: ContinuousXtoC와 CtoZ의 레이어 개수는 동일해야 함 (원래 논문의 구조 유지)
class PartialVAE(nn.Module):
    '''
    x -> c -> z -> h -> x
    '''
    def __init__(self,
                 input_type, # input의 type
                 num_con_features, # con_X의 개수
                 num_cat_features, # cat_X의 개수
                 hidden_dim_con, # con_x_to_c의 MLP layer의 hidden layer의 뉴런 개수 
                 most_categories, # 가장 많은 범주를 가지고 있는 feature의 범주의 수
                 c_dim, # c의 dimension
                 hid_enc, # CtoZ의 hidden dim
                 hid_dec, # ZtoH의 hidden dim
                 latent_dim): # z의 dimension
        super().__init__()
        self.input_type = input_type # mixed, continuous, categorical
        self.num_con_features = num_con_features
        self.num_cat_features = num_cat_features
        self.most_categories = most_categories
        self.c_dim = c_dim

        if self.input_type == "mixed":
            self.last_con_index = num_con_features

        if self.input_type in ("mixed", "continuous"):
            self.con_x_to_c = ContinuousXtoC(
                num_con_features=num_con_features,
                hidden_dim=hidden_dim_con,
                c_dim=c_dim
            )

        if self.input_type in ("mixed", "categorical"):
            self.cat_x_to_c = CategoricalXtoC(
                num_cat_features=num_cat_features,
                most_categories=most_categories,
                c_dim=c_dim
            )

        self.c_to_z = CtoZ(in_dim=c_dim, hidden_dim=hid_enc, out_dim=latent_dim)
        self.z_to_h = ZtoH(latent_dim=latent_dim, hidden_dim=hid_dec)
        self.h_to_x = nn.Linear(hid_dec, num_con_features + num_cat_features * most_categories)
        
    def x_to_c(self, x, m):
        if self.input_type == "continuous":
            return self.con_x_to_c(x, m)

        elif self.input_type == "categorical":
            return self.cat_x_to_c(x, m)

        elif self.input_type == "mixed": # con -> cat feature 순으로 입력이 들어와야 함
            x_con   = x[:, :self.last_con_index]
            m_con   = m[:, :self.last_con_index]
            x_cat   = x[:, self.last_con_index:]
            m_cat   = m[:, self.last_con_index:]

            c_con = self.con_x_to_c(x_con, m_con) if self.con_x_to_c else 0
            c_cat = self.cat_x_to_c(x_cat, m_cat) if self.cat_x_to_c else 0
            return c_con + c_cat # 전체 feature의 c로의 임베딩 출력

    @staticmethod
    def reparameterize(mu, sig, n_samples): # 학습 시 샘플 1개, inference시 샘플 5 ~ 10개 
        if n_samples == 1:
            eps = torch.randn_like(mu)
            return mu + sig * eps
        else:
            z = torch.distributions.Normal(mu, sig).rsample((n_samples,))
            return z

    def forward(self, x, m, n_samples: int = 1):
        c = self.x_to_c(x, m) # x -> c
        mu, sig = self.c_to_z(c) # c -> z (mu, sig)
        if n_samples == 1:
            z = self.reparameterize(mu, sig, 1)
            h = self.z_to_h(z) # 1개 sampling 후 z -> h
            x_hat = self.h_to_x(h) # h -> x복원
        else:
            z = self.reparameterize(mu, sig, n_samples)
            SB, D = z.shape[0]*z.shape[1], z.shape[2]
            z = z.reshape(SB, D)
            h = self.z_to_h(z) # n개 sampling 후 z -> h
            x_hat = self.h_to_x(h).view(n_samples, x.size(0), -1) # h -> x복원
        return x_hat, (mu, sig)


    # loss 관련 메서드
    def reconstruction_nll(self, x, m, x_hat, obs_sigma=1.0):
        Dcon = self.num_con_features
        Dcat = self.num_cat_features
        Cmax = self.most_categories

        nll = 0.0

        if Dcon > 0:
            x_con_true = x[:, :Dcon].float()
            m_con = m[:, :Dcon].float()
            x_con_pred = x_hat[:, :Dcon] # 연속부는 평균값으로 해석
            nll = nll + nll_continuous(x_con_true, x_con_pred, m_con, obs_sigma)

        if Dcat > 0:
            x_cat_true = x[:, Dcon:Dcon+Dcat].long() # 각 feature의 카테고리 index
            m_cat = m[:, Dcon:Dcon+Dcat].float()
            logits_cat = x_hat[:, Dcon:Dcon + Dcat*Cmax]
            nll = nll + nll_categorical(x_cat_true, logits_cat, m_cat)

        return nll

    def loss_func(self, x, m, obs_sigma=1.0, n_samples=1):
        x_hat, (mu, sig) = self.forward(x, m, n_samples=1)
        nll_x = self.reconstruction_nll(x, m, x_hat, obs_sigma=obs_sigma)
        kl = kl_01_loss(mu, sig)
        return nll_x + kl, {"KL": kl.detach(), "NLL_X": nll_x.detach()}
    