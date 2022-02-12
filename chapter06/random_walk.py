from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

START = 3
GOALS = [0, 6]

PROB_LEFT = 0.5
ACTION_LEFT = -1
ACTION_RIGHT = 1

# true state value from bellman equation
TRUE_VALUES = np.linspace(0, 1, 7)


def choose_action() -> int:
    """Choose an action randomly."""
    if np.random.random() < PROB_LEFT:
        return ACTION_LEFT
    return ACTION_RIGHT


def step(state: int, action: int) -> Tuple[int]:
    """Get reward and next state given current state and action."""
    next_state = state + action
    if next_state == 6:
        reward = 1
    else:
        reward = 0
    return next_state, reward


def temporal_difference(state_values: np.ndarray, alpha: float = 0.1, gamma: float = 1.) -> None:
    """
    Policy is proceeding either left or right by one state on each step, with equal probability.
    """
    state = START
    while True:
        action = choose_action()
        next_state, reward = step(state, action)

        # TD update
        state_values[state] += alpha * (reward + gamma * state_values[next_state] - state_values[state])

        if next_state in GOALS:
            break
        state = next_state


def monte_carlo(state_values: np.ndarray, alpha: float = 0.1) -> None:
    # TODO
    state = START
    trajectory = [state]
    while True:
        action = choose_action()
        state, reward = step(state, action)
        trajectory.append(state)

        if state in GOALS:
            G = reward
            break

    for s in trajectory[:-1]:
        state_values[s] += alpha * (G - state_values[s])


def example_6_2():
    # Left figure
    state_values = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])

    episodes = [0, 1, 10, 100]
    values_to_plot = []
    for i in range(episodes[-1] + 1):
        if i in episodes:
            values_to_plot.append(state_values.copy())

        temporal_difference(state_values)

    x = ["A", "B", "C", "D", "E"]
    plt.plot(x, TRUE_VALUES[1:6], marker="o", label="True values")
    for i, v in zip(episodes, values_to_plot):
        plt.plot(x, v[1:6], marker="o", label=f"{i} episodes")
    plt.xlabel("State")
    plt.ylabel("Estimated Value")
    plt.legend()
    plt.show()

    # Right figure
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    runs = 100
    episodes = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        if i < len(td_alphas):
            method = "TD"
            linestyle = "solid"
        else:
            method = "MC"
            linestyle = "dashdot"

        total_errors = np.zeros(episodes + 1)
        for _ in range(runs):
            errors = []
            state_values = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])
            for _ in range(episodes + 1):
                errors.append(np.linalg.norm(state_values[1:6] - TRUE_VALUES[1:6]))
                if method == "TD":
                    temporal_difference(state_values, alpha=alpha)
                else:
                    monte_carlo(state_values, alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs

        plt.plot(total_errors, linestyle=linestyle, label=f"{method}, $\\alpha = {alpha:.2f}$")
    plt.xlabel("Walks/Episodes")
    plt.ylabel("Empirical RMS error, averaged over states")
    plt.legend()
    plt.show()
