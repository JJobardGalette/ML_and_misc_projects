"""UCB algorithm with functions to display what's happening cf. Auer et al. 2002"""
# %%
from environment import Environment
import numpy as np
import matplotlib.pyplot as plt

class UCB:
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
                self.b_a[arm] = self.empirical_means[arm] + np.sqrt(2*np.log(iteration+1)/len(self.rewards_list[arm]))
        self.b_a_history.append(self.b_a.copy())
        self.empirical_means_history.append(self.empirical_means.copy())

    def reset_bandits(self):
        self.empirical_means = np.ones(self.env.arms_number)*float('inf')
        self.rewards_list = [[] for _ in range(self.env.arms_number)]
        self.b_a = np.array([float('inf')]*self.env.arms_number)
        self.arm_history = []
        self.b_a_history = []

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
        theoretical_regret = 9 * np.sqrt(self.env.arms_number * count[1:] * np.log(count[1:])) + 8*self.env.arms_number/3
        plt.plot(count[1:], theoretical_regret, linestyle='--')
        plt.text(0, 1, "as one can see, this bound is not the best")
        plt.show()

if __name__ == '__main__':
    env = Environment([0.6, 0.5, 0.8, 0.9, 0.3, 0.31])
    ucb = UCB(env)
    ucb.run(10000)
    ucb.plot_convergence_result()
    ucb.plot_regret()
