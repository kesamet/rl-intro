import itertools
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

WORLD_LENGTH = 1002
START = 500
GOALS = [0, 1001]
STATES = np.arange(1, WORLD_LENGTH - 1)

PROB_LEFT = 0.5
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS  = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100

# true state value from bellman equation
TRUE_VALUES = np.linspace(-1, 1, WORLD_LENGTH)
TRUE_VALUES[0] = TRUE_VALUES[-1] = 0


def dynamic_program(inplace: bool = True, gamma: float = 1., theta: float = 1e-2) -> np.ndarray:
    """4.1 Iterative policy evaluation (DP)."""
    def _step(state: int, step: int) -> Tuple[int]:
        """Get reward and next state given current state and action."""
        next_state = max(min(state + step, WORLD_LENGTH - 1), 0)            
        if next_state == WORLD_LENGTH - 1:
            reward = 1
        elif next_state == 0:
            reward = -1
        else:
            reward = 0
        return next_state, reward

    # Using guess in order to converge faster
    # state_values = np.zeros(WORLD_LENGTH)
    state_values = np.linspace(-1, 1, WORLD_LENGTH)
    state_values[0] = 0
    state_values[-1] = 0
    prob = 0.5 / STEP_RANGE
    iters = 0
    while True:
        old_state_values = state_values.copy()

        for i in STATES:
            _v = 0
            for sgn, size in itertools.product(ACTIONS, np.arange(1, STEP_RANGE + 1)):
                next_i, reward = _step(i, sgn * size)
                if inplace:
                    # Asynchronous/inplace
                    _v += prob * (reward + gamma * state_values[next_i])
                else:
                    # Synchronous
                    _v += prob * (reward + gamma * old_state_values[next_i])
            state_values[i] = _v

        if np.abs(state_values - old_state_values).max() < theta:
            break

        iters += 1

    print(f"  {iters} iterations")
    return state_values


def step(state: int, action: int) -> Tuple[int]:
    """Get reward and next state given current state and action."""
    next_state = max(min(state + action, WORLD_LENGTH - 1), 0)
    if next_state == WORLD_LENGTH - 1:
        reward = 1
    elif next_state == 0:
        reward = -1
    else:
        reward = 0
    return next_state, reward


def choose_action() -> int:
    """Choose an action randomly."""
    if np.random.binomial(1, PROB_LEFT) == 1:
        sgn = ACTION_LEFT
    else:
        sgn = ACTION_RIGHT
    return sgn * np.random.randint(1, STEP_RANGE + 1)


class ValueFunction:
    def __init__(self, n_states: int, n_groups: int):
        self.n_states = n_states
        self.n_groups = n_groups  # num of aggregations
        self.group_size = n_states // n_groups
        self.thetas = np.zeros(n_groups)

    def value(self, state):
        """Get state value."""
        if state in GOALS:
            return 0
        return self.thetas[(state - 1) // self.group_size]

    def update(self, delta, state):
        """Update thetas."""
        self.thetas[(state - 1) // self.group_size] += delta


def gradient_montecarlo(
    value_function: np.ndarray, distribution: int, alpha: float, gamma: float = 1.
) -> None:
    # TODO: incorporate gamma
    state = START

    # Track states, rewards and time
    trajectory = [state]
    # rewards = [0]
    while True:
        action = choose_action()
        state, reward = step(state, action)
        trajectory.append(state)
        # rewards.append(reward)

        if state in GOALS:
            G = reward
            break

    for state in trajectory[:-1]:
        delta = alpha * (G - value_function.value(state))
        value_function.update(delta, state)
        if distribution is not None:
            distribution[state] += 1


def figure_9_1():
    print("Compute state values using DP")
    start = time.time()
    true_values = dynamic_program()
    print(f"  Time taken = {time.time() - start:.0f} secs")

    episodes = int(1e5)
    alpha = 2e-5
    n_groups = 10  # 10 aggregations

    print("Compute state values using gradient Monte-Carlo")
    start = time.time()
    value_function = ValueFunction(WORLD_LENGTH - len(GOALS), n_groups)
    distribution = np.zeros(WORLD_LENGTH)
    for ep in range(episodes):
        if ep % 10000 == 0:
            print(f"  Episode {ep}")
        gradient_montecarlo(value_function, distribution, alpha)

    distribution /= np.sum(distribution)
    state_values = [value_function.value(i) for i in STATES]
    print(f"  Time taken = {time.time() - start:.0f} secs")

    plt.figure(figsize=(8, 10))

    plt.subplot(2, 1, 1)
    plt.plot(STATES, state_values, label="Approximate MC value")
    plt.plot(STATES, true_values[1: -1], label="True value")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(STATES, distribution[1: -1], label="State distribution")
    plt.xlabel("State")
    plt.ylabel("Distribution")
    plt.legend()
    plt.show()


# def policy_action(
#     q_values: np.ndarray,
#     state: int,
#     epsilon: float,
# ) -> np.ndarray:
#     """Choose an action based on epsilon-greedy algorithm."""
#     if np.random.binomial(1, epsilon) == 1:
#         return np.random.choice(ACTIONS)

#     values_ = q_values[state, :]
#     return np.random.choice(np.array(ACTIONS)[values_ == np.max(values_)])


# def sarsa(q_values: np.ndarray, n: int, alpha: float, epsilon: float = 0.1, gamma: float = 1.) -> None:
#     state = START
#     action = policy_action(q_values, state, epsilon)

#     # Track states, actions, rewards and time
#     states = [state]
#     actions = [action]
#     rewards = [0]
#     t = 0

#     T = float("inf")

#     while True:
#         if t < T:
#             next_state, reward = step(state, action)
#             states.append(next_state)
#             rewards.append(reward)
#             if next_state in GOALS:
#                 T = t + 1
#             else:
#                 next_action = policy_action(q_values, state, epsilon)
#                 actions.append(next_action)

#         # n-step Sarsa update
#         tau = t - n + 1
#         if tau >= 0:
#             G = sum([
#                 gamma ** (i - tau - 1) * rewards[i]
#                 for i in range(tau + 1, min(tau + n, T) + 1)
#             ])
#             if tau + n < T:
#                 G += gamma ** n * q_values[states[tau + n], actions[tau + n]]
#             q_values[states[tau], actions[tau]] += alpha * (G - q_values[states[tau], actions[tau]])

#         if tau == T - 1:
#             break
#         state = next_state
#         action = next_action
#         t += 1


# def figure_7_2():
#     alphas = np.linspace(0, 1, 11)
#     ns = 2 ** np.arange(0, 10)
#     runs = 100
#     episodes = 10

#     errors = np.zeros((len(ns), len(alphas)))
#     for i, n in enumerate(ns):
#         for j, alpha in enumerate(alphas):
#             for _ in range(runs):
#                 state_values = np.zeros(WORLD_LENGTH)
#                 for _ in range(episodes):
#                     temporal_difference(state_values, n, alpha)

#                     rmse = np.sqrt(np.mean((state_values[1:20] - TRUE_VALUES[1:20]) ** 2))
#                     errors[i, j] += rmse

#     errors /= (runs * episodes)

#     for i, n in enumerate(ns):
#         plt.plot(alphas, errors[i, :], label=f"n = {n}")
#     plt.xlabel("$\\alpha$")
#     plt.ylabel("RMS error")
#     plt.ylim([0.25, 0.55])
#     plt.legend()
#     plt.show()


def test_sarsa():
    alpha = 0.5
    n = 2
    episodes = 10

    q_values = np.zeros((WORLD_LENGTH, len(ACTIONS)))
    for _ in range(episodes):
        sarsa(q_values, n, alpha)
