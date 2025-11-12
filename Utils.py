import numpy as np
import torch
import random

class Utils():
    def __init__(self):
        pass # util 메서드들은 생성자 X

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
