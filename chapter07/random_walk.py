from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

WORLD_LENGTH = 21
START = 10
GOALS = [0, 20]

PROB_LEFT = 0.5
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS  = [ACTION_LEFT, ACTION_RIGHT]

# true state value from bellman equation
TRUE_VALUES = np.linspace(-1, 1, WORLD_LENGTH)
TRUE_VALUES[0] = TRUE_VALUES[-1] = 0


def step(state: int, action: int) -> Tuple[int]:
    """Get reward and next state given current state and action."""
    next_state = state + action
    if next_state == 20:
        reward = 1
    elif next_state == 0:
        reward = -1
    else:
        reward = 0
    return next_state, reward


def choose_action() -> int:
    """Choose an action randomly."""
    if np.random.random() < PROB_LEFT:
        return ACTION_LEFT
    return ACTION_RIGHT


def temporal_difference(state_values: np.ndarray, n: int, alpha: float, gamma: float = 1.) -> None:
    state = START

    # Track states, rewards and time
    states = [state]
    rewards = [0]
    t = 0

    T = float("inf")

    while True:
        if t < T:
            action = choose_action()
            next_state, reward = step(state, action)
            states.append(next_state)
            rewards.append(reward)
            if next_state in GOALS:
                T = t + 1

        # n-step TD update
        tau = t - n + 1
        if tau >= 0:
            G = sum([
                gamma ** (i - tau - 1) * rewards[i]
                for i in range(tau + 1, min(tau + n, T) + 1)
            ])
            if tau + n < T:
                G += gamma ** n * state_values[states[tau + n]]
            state_values[states[tau]] += alpha * (G - state_values[states[tau]])

        if tau == T - 1:
            break
        state = next_state
        t += 1


def policy_action(
    q_values: np.ndarray,
    state: int,
    epsilon: float,
) -> np.ndarray:
    """Choose an action based on epsilon-greedy algorithm."""
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(ACTIONS)

    values_ = q_values[state, :]
    return np.random.choice(np.array(ACTIONS)[values_ == np.max(values_)])


def sarsa(q_values: np.ndarray, n: int, alpha: float, epsilon: float = 0.1, gamma: float = 1.) -> None:
    state = START
    action = policy_action(q_values, state, epsilon)

    # Track states, actions, rewards and time
    states = [state]
    actions = [action]
    rewards = [0]
    t = 0

    T = float("inf")

    while True:
        if t < T:
            next_state, reward = step(state, action)
            states.append(next_state)
            rewards.append(reward)
            if next_state in GOALS:
                T = t + 1
            else:
                next_action = policy_action(q_values, state, epsilon)
                actions.append(next_action)

        # n-step Sarsa update
        tau = t - n + 1
        if tau >= 0:
            G = sum([
                gamma ** (i - tau - 1) * rewards[i]
                for i in range(tau + 1, min(tau + n, T) + 1)
            ])
            if tau + n < T:
                G += gamma ** n * q_values[states[tau + n], actions[tau + n]]
            q_values[states[tau], actions[tau]] += alpha * (G - q_values[states[tau], actions[tau]])

        if tau == T - 1:
            break
        state = next_state
        action = next_action
        t += 1


def figure_7_2():
    alphas = np.linspace(0, 1, 11)
    ns = 2 ** np.arange(0, 10)
    runs = 100
    episodes = 10

    errors = np.zeros((len(ns), len(alphas)))
    for i, n in enumerate(ns):
        for j, alpha in enumerate(alphas):
            for _ in range(runs):
                state_values = np.zeros(WORLD_LENGTH)
                for _ in range(episodes):
                    temporal_difference(state_values, n, alpha)

                    rmse = np.sqrt(np.mean((state_values[1:20] - TRUE_VALUES[1:20]) ** 2))
                    errors[i, j] += rmse

    errors /= (runs * episodes)

    for i, n in enumerate(ns):
        plt.plot(alphas, errors[i, :], label=f"n = {n}")
    plt.xlabel("$\\alpha$")
    plt.ylabel("RMS error")
    plt.ylim([0.25, 0.55])
    plt.legend()
    plt.show()


def _test_sarsa():
    alpha = 0.5
    n = 2
    episodes = 10

    q_values = np.zeros((WORLD_LENGTH, len(ACTIONS)))
    for _ in range(episodes):
        sarsa(q_values, n, alpha)
