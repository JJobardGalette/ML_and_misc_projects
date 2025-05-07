"""KL-UCB algorithm with functions to display what's happening cf. Garivier -- Cappé 2011"""
# %%
from environment import Environment
import numpy as np
import matplotlib.pyplot as plt
import scipy

def confidence_level(x):
    if x == 1:
        return 1e20
    return np.log(x)+3*np.log(np.log(x))

def kl_div(a, b):
    return a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))

def optimisation(n, t, theta, bounds=(1e-10, 1-1e-10)):
    def objective(q):
        if n*kl_div(theta, q) < confidence_level(t):
            return -q
        else:
            return 1e6
    optimisation_result = scipy.optimize.minimize_scalar(objective, bounds=bounds, method='bounded')
    if not optimisation_result.success:
        raise RuntimeError("optimisation failed!")
    else:
        return optimisation_result.x

class KLUCB:
    def __init__(self, env:Environment):
        self.env = env
        self.empirical_means = np.ones(self.env.arms_number)*float('inf')
        self.rewards_list = [[] for _ in range(self.env.arms_number)]
        self.b_a = np.array([float('inf')]*self.env.arms_number)
        self.arm_history = []
        self.b_a_history = []
        self.empirical_means_history = []
        self.regret_history = [0]
        self.best_arm = np.argmax(env.means)
        self.best_reward = max(self.env.means)

    def run(self, iterations):
        for i in range(iterations):
            arm = np.argmax(self.b_a)
            reward = self.env.play_arm(arm)
            self.regret_history.append(self.regret_history[-1] + env.means[int(self.best_arm)] - env.means[int(arm)])
            self.arm_history.append(arm)
            self.rewards_list[arm].append(reward)
            self.compute_empirical_means_and_b_a(i)

    def compute_empirical_means_and_b_a(self, iteration):
        for arm in range(self.env.arms_number):
            if self.rewards_list[arm]:
                self.empirical_means[arm] = np.mean(self.rewards_list[arm])
                n_arm = len(self.rewards_list[arm])
                self.b_a[arm] = optimisation(n_arm, self.env.time, self.empirical_means[arm])
        self.b_a_history.append(self.b_a.copy())
        self.empirical_means_history.append(self.empirical_means.copy())

    def reset_bandits(self):
        self.empirical_means = np.ones(self.env.arms_number)*float('inf')
        self.rewards_list = [[] for _ in range(self.env.arms_number)]
        self.b_a = np.array([float('inf')]*self.env.arms_number)
        self.arm_history = []
        self.b_a_history = []
    
    def compute_regret(self, count):
        arm_factor = 0
        for arm in self.env.means:
            if arm != self.best_reward:
                arm_factor += (self.best_reward - arm)/kl_div(arm, self.best_reward)
        return np.log(count[1:])*arm_factor

    def plot_convergence_result(self):
        plt.plot(self.empirical_means_history)
        plt.ylim((0, 1.1))
        incertitudes = np.array(self.b_a_history) - np.array(self.empirical_means_history)
        n_rows, n_cols = incertitudes.shape
        count = np.arange(0, n_rows)
        for i in range(n_cols):
            plt.fill_between(count,
                (np.array(self.empirical_means_history)-incertitudes)[:, i],
                (np.array(self.empirical_means_history) + incertitudes)[:, i], alpha=0.2)
        plt.show()

    def plot_regret(self):
        count = np.arange(0, len(self.arm_history))
        plt.plot(count, self.regret_history[1:])
        theoretical_regret = self.compute_regret(count)
        plt.plot(count[1:], theoretical_regret, linestyle='--')
        # plt.text(0, 1, "as one can see, this bound is not the best")
        plt.show()

if __name__ == '__main__':
    env = Environment([0.6, 0.4])
    klucb = KLUCB(env)
    klucb.run(10000)
    klucb.plot_convergence_result()
    klucb.plot_regret()
