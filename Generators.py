import numpy as np

# uniform에서 뽑는 것은 최대, 최솟값의 위치가 애매해져서 쓰지 않았음

class GaussianSampler:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def generate(self, x, m):
        gaussian_samples = np.random.normal(
            loc=self.mean,
            scale=self.std,
            size=x.shape
        )

        return m * x + (1 - m) * gaussian_samples