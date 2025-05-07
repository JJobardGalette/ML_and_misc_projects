"""Simple environment with bernoulli random rewards
No Assumptions is made yet on which arm is optimal"""

import numpy as np

class Environment:
    def __init__(self, means):
        self.means = means
        self.arms_number = len(self.means)

    def play_arm(self, i):
        return np.random.random() < self.means[i]
