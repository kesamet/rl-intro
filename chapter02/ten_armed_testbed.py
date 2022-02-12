# TODO: gradient
from typing import List

import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(
        self, k: int = 10, epsilon: float = 0., q_initial:float = 0., step_size: float = None,
        true_reward: float = 0,
        update_method: str = "constant_step",  # constant_step, sample_averages
        choose_method: str = None,  # Non=greedy, ucb, gradient
        ucb_coef: float = None,
        gradient_baseline: bool = None,
    ):
        self.k = k
        self.actions = np.arange(k)
        self.epsilon = epsilon
        self.q_initial = q_initial
        self.step_size = step_size
        self.true_reward = true_reward

        self.update_method = update_method

        self.choose_method = choose_method
        self.ucb_coef = ucb_coef
        if choose_method == "ucb":
            assert self.ucb_coef is not None
        self.gradient_baseline = gradient_baseline

    def reset(self) -> None:
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.best_action = np.argmax(self.q_true)
        self.time = 0
        self.q = np.zeros(self.k) + self.q_initial
        self.action_count = np.zeros(self.k)

        self.average_reward = None

    def choose(self) -> int:
        if self.choose_method == "gradient":
            exp_q = np.exp(self.q)
            self.action_prob = exp_q / np.sum(exp_q)
            return np.random.choice(self.actions, p=self.action_prob)

        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(self.actions)

        if self.choose_method == "ucb":
            q_ = (
                self.q
                + self.ucb_coef * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            )
        else:
            q_ = self.q

        q_max = np.max(q_)
        return np.random.choice(np.where(q_ == q_max)[0])

    def step(self, action: int) -> float:
        self.time += 1
        reward = self.q_true[action] + np.random.randn()
        self.action_count[action] += 1
        
        if self.update_method  == "sample_averages":
            # update estimation using sample averages
            self.q[action] += (reward - self.q[action]) / self.action_count[action]
        elif self.update_method == "constant_step":
            # update estimation with constant step size
            self.q[action] += self.step_size * (reward - self.q[action])
        elif self.choose_method == "gradient":
            if self.time == 1:
                self.average_reward = reward
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            self.q += (
                self.step_size
                * (reward - self.average_reward * self.gradient_baseline)
                * (one_hot - self.action_prob)
            )
            self.average_reward += (reward - self.average_reward) / self.time
        else:
            assert False

        return reward


def simulate(runs: int, time_steps: int, bandits: List[Bandit]):
    rewards = np.zeros((len(bandits), runs, time_steps))
    best_action_counts = np.zeros_like(rewards)
    for i, bandit in enumerate(bandits):
        print(f"Bandit {i}")
        for r in range(runs):
            if r % 1000 == 0:
                print(f"  Run {r}")
            bandit.reset()
            for t in range(time_steps):
                action = bandit.choose()
                reward = bandit.step(action)
                rewards[i, r, t] = reward

                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1

    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def figure_2_2():
    runs = 2000
    time_steps = 1000
    epsilons = [0.1, 0.01, 0]
    bandits = [Bandit(epsilon=eps, update_method="sample_averages") for eps in epsilons]
    mean_best_action_counts, mean_rewards = simulate(runs, time_steps, bandits)

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, mean_rewards):
        plt.plot(rewards, label=f"$\epsilon = {eps}$")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, mean_best_action_counts):
        plt.plot(counts, label=f"$\epsilon = {eps}$")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.show()


def figure_2_3():
    runs = 2000
    time_steps = 1000

    bandits = [
        Bandit(epsilon=0, q_initial=5, update_method="constant_step", step_size=0.1),
        Bandit(epsilon=0.1, q_initial=0, update_method="constant_step", step_size=0.1),
    ]
    pct_optimal_action, _ = simulate(runs, time_steps, bandits)

    plt.plot(pct_optimal_action[0], label="Optimistic, greedy\n$Q_1=5, \epsilon = 0$")
    plt.plot(pct_optimal_action[1], label="Realistic, $\epsilon$-greedy\n$Q_1=5, \epsilon = 0.1$")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.show()


def figure_2_4():
    runs = 2000
    time_steps = 1000
    bandits = [
        Bandit(epsilon=0, ucb_coef=2, update_method="sample_averages"),
        Bandit(epsilon=0.1, update_method="sample_averages"),
    ]
    _, average_rewards = simulate(runs, time_steps, bandits)

    plt.plot(average_rewards[0], label="UCB, $c = 2$")
    plt.plot(average_rewards[1], label="$\epsilon$-greedy $\epsilon = 0.1$")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()


def figure_2_5():
    runs = 2000
    time_steps = 1000
    bandits = [
        Bandit(step_size=0.1, true_reward=4, choose_method="gradient", gradient_baseline=True),
        Bandit(step_size=0.1, true_reward=4, choose_method="gradient", gradient_baseline=False),
        Bandit(step_size=0.4, true_reward=4, choose_method="gradient", gradient_baseline=True),
        Bandit(step_size=0.4, true_reward=4, choose_method="gradient", gradient_baseline=False),
    ]
    pct_optimal_action, _ = simulate(runs, time_steps, bandits)

    labels = [
        "$\\alpha = 0.1$, with baseline",
        "$\\alpha = 0.1$, without baseline",
        "$\\alpha = 0.4$, with baseline",
        "$\\alpha = 0.4$, without baseline",
    ]
    for i in range(len(bandits)):
        plt.plot(pct_optimal_action[i], label=labels[i])
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
