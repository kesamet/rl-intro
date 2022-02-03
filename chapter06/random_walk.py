import numpy as np
import matplotlib.pyplot as plt

TRUE_VALUES = np.linspace(0, 1, 7)

PROB_LEFT = 0.5
ACTION_LEFT = -1
ACTION_RIGHT = 1


def temporal_difference(values, start_state=3, alpha=0.1, gamma=1.):
    """
    Policy is proceeding either left or right by one state on each step, with equal probability.
    """
    state = start_state
    while True:
        if np.random.random() < PROB_LEFT:
            action = ACTION_LEFT
        else:
            action = ACTION_RIGHT
        new_state = state + action
        if new_state == 6:
            reward = 1
        else:
            reward = 0

        # TD update
        values[state] += alpha * (reward + gamma * values[new_state] - values[state])
        state = new_state
        if state in [0, 6]:
            break


def monte_carlo(values, start_state=3, alpha=0.1):
    state = start_state
    trajectory = [state]
    while True:
        if np.random.random() < PROB_LEFT:
            action = ACTION_LEFT
        else:
            action = ACTION_RIGHT
        state += action
        trajectory.append(state)
        if state == 6:
            returns = 1
            break
        elif state == 0:
            returns = 0
            break

    for s in trajectory[:-1]:
        values[s] += alpha * (returns - values[s])


def compute_state_values():
    """Example 6.2 left."""
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


def rms_error():
    """Example 6.2 right."""
    # Same alpha value can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes + 1)
        if i < len(td_alphas):
            method = "TD"
            linestyle = "solid"
        else:
            method = "MC"
            linestyle = "dashdot"
        for _ in range(runs):
            errors = []
            state_values = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])
            for i in range(episodes + 1):
                errors.append(np.linalg.norm(TRUE_VALUES - state_values))
                if method == "TD":
                    temporal_difference(state_values, alpha=alpha)
                else:
                    monte_carlo(state_values, alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=f"{method}, $\\alpha$ = {alpha:.2f}")
    plt.xlabel("Walks/Episodes")
    plt.ylabel("Empirical RMS error, averaged over states")
    plt.legend()
    plt.show()
