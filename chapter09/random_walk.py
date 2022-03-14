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


def step(state: int, action: int) -> Tuple[int]:
    """Get reward and next state given current state and action."""
    step = action * np.random.randint(1, STEP_RANGE + 1)
    next_state = max(min(state + step, WORLD_LENGTH - 1), 0)
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
        return ACTION_LEFT
    return ACTION_RIGHT


class ValueFunction:
    def __init__(self, n_states: int, num_of_groups: int):
    # @num_of_groups: # of aggregations
        self.n_states = n_states
        self.num_of_groups = num_of_groups
        self.group_size = n_states // num_of_groups
        self.thetas = np.zeros(num_of_groups)

    def value(self, state):
        """Get state value."""
        if state in GOALS:
            return 0

        return self.thetas[(state - 1) // self.group_size]

    def update(self, delta, state):
        """Update thetas."""
        self.thetas[(state - 1) // self.group_size] += delta


def gradient_montecarlo(value_function: np.ndarray, distribution: int, alpha: float, gamma: float = 1.) -> None:
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
    episodes = int(1e5)
    alpha = 2e-5

    # we have 10 aggregations in this example, each has 100 states
    value_function = ValueFunction(WORLD_LENGTH - len(GOALS), 10)
    distribution = np.zeros(WORLD_LENGTH)
    for _ in range(episodes):
        gradient_montecarlo(value_function, alpha, distribution)

    distribution /= np.sum(distribution)
    state_values = [value_function.value(i) for i in STATES]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(STATES, state_values, label='Approximate MC value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(STATES, distribution[1: -1], label='State distribution')
    plt.xlabel('State')
    plt.ylabel('Distribution')
    plt.legend()

    plt.savefig('../images/figure_9_1.png')
    plt.close()


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
