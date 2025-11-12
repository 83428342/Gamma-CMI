import numpy as np
import torch
import torch.nn.functional as F

class FeatureAcquisition():
    def __init__(self, x, m, generative_model, num_samples, predictor, alpha=1, gamma=0):
        self.x = x
        self.m = m
        self.generative_model = generative_model
        self.num_samples = num_samples
        self.predictor = predictor
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
        
    def conditional_sample(self):
        x = self.x
        m = self.m
        num_samples = self.num_samples
        x_rep = np.repeat(a=x, repeats=num_samples, axis=0)
        m_rep = np.repeat(a=m, repeats=num_samples, axis=0)

        x_sampled = self.generative_model.generate(x_rep, m_rep)

        return x_sampled

    def alpha_gamma_cmi(self):
        x = self.x
        m = self.m
        gamma = self.gamma
        num_samples = self.num_samples
        predictor = self.predictor

        device = next(predictor.parameters()).device

        x_sampled = self.conditional_sample() # 전체 값이 샘플링된 x
        m_repeated = np.repeat(a=m, repeats=num_samples, axis=0) # 기존 x의 mask

        m_upsampled = np.random.binomial(n=1, p=gamma, size=m_repeated.shape) # 각 feature별로 0 또는 1로 변형 
        m_repeated = np.maximum(m_repeated, m_upsampled) # 위에서는 모든 feature별로 진행했으니 max로 병합
        
        x_sampled = torch.tensor(x_sampled, dtype=torch.float32, device=device) # tensor로 변경
        m_repeated = torch.tensor(m_repeated, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = predictor(x_sampled, m_repeated)
            p_now = torch.softmax(logits, dim=-1).cpu().numpy()
        h_now = self.entropy(p=p_now)

        out = []

        for f in range(x.shape[-1]):
            m_copy = m_repeated.clone() # numpy의 copy method
            m_copy[:, f] = 1.0

            with torch.no_grad():
                logits = predictor(x_sampled, m_copy)
                p_with = torch.softmax(logits, dim=-1).cpu().numpy()
            h_with = self.entropy(p=p_with)

            entropy_diff = h_now - h_with
            entropy_diff = entropy_diff.reshape(-1, num_samples).mean(-1)
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