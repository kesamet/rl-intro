import itertools
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

WORLD_LENGTH = 1002
START = 500
GOALS = [0, 1001]
STATES = np.arange(1, WORLD_LENGTH - 1)
N_STATES = len(STATES)

PROB_LEFT = 0.5
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS_DIR = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action direction
STEP_RANGE = 100


def dynamic_program(
    inplace: bool = True, gamma: float = 1.0, theta: float = 1e-2
) -> np.ndarray:
    """4.1 Iterative policy evaluation (DP)."""
    print("Compute true state values using DP")

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
            for sgn, size in itertools.product(
                ACTIONS_DIR, np.arange(1, STEP_RANGE + 1)
            ):
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
    return state_values[1:-1]


TRUE_VALUES = dynamic_program()


class ValueFunction:
    def __init__(self, n_states: int, n_groups: int):
        self.n_states = n_states
        self.n_groups = n_groups  # num of aggregations
        self.group_size = n_states // n_groups
        self.thetas = np.zeros(n_groups)

    def value(self, state: int) -> float:
        """Get state value."""
        if state in GOALS:
            return 0
        return self.thetas[(state - 1) // self.group_size]

    def update(self, delta: float, state: int) -> None:
        """Update thetas."""
        self.thetas[(state - 1) // self.group_size] += delta


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


def gradient_montecarlo(
    value_function: np.ndarray,
    alpha: float,
    gamma: float = 1.0,
    distribution: np.ndarray = None,
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
            G = reward  # TODO
            break

    for state in trajectory[:-1]:
        delta = alpha * (G - value_function.value(state))
        value_function.update(delta, state)
        if distribution is not None:
            distribution[state] += 1


def semi_gradient_td(
    value_function: np.ndarray,
    alpha: float,
    n: int,
    gamma: float = 1.0,
) -> None:
    state = START

    # Track states, actions, rewards and time
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
            G = sum(
                [
                    gamma ** (i - tau - 1) * rewards[i]
                    for i in range(tau + 1, min(tau + n, T) + 1)
                ]
            )
            if tau + n < T:
                G += gamma**n * value_function.value(states[tau + n])
            state_tau = states[tau]
            if state_tau not in GOALS:
                delta = alpha * (G - value_function.value(state_tau))
                value_function.update(delta, state_tau)

        if tau == T - 1:
            break
        state = next_state
        t += 1


def figure_9_1():
    """Gradient Monte-Carlo."""
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
        gradient_montecarlo(value_function, alpha, distribution=distribution)
    print(f"  Time taken = {time.time() - start:.0f} secs")

    state_values = [value_function.value(i) for i in STATES]
    plt.plot(STATES, state_values, label="Approximate MC value")
    plt.plot(STATES, TRUE_VALUES, label="True value")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    distribution /= np.sum(distribution)
    plt.plot(STATES, distribution[1:-1], label="State distribution")
    plt.xlabel("State")
    plt.ylabel("Distribution")
    plt.legend()
    plt.show()


def figure_9_2a():
    """Semi-gradient temporal difference."""
    print("Compute state values using semi-gradient temporal difference")
    start = time.time()
    episodes = int(1e5)
    alpha = 2e-4
    n = 1
    n_groups = 10
    value_function = ValueFunction(N_STATES, n_groups)
    for ep in range(episodes):
        if ep % 10000 == 0:
            print(f"  Episode {ep}")
        semi_gradient_td(value_function, alpha, n)
    print(f"  Time taken = {time.time() - start:.0f} secs")

    state_values = [value_function.value(i) for i in STATES]
    plt.plot(STATES, state_values, label="Approximate TD value")
    plt.plot(STATES, TRUE_VALUES, label="True value")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def figure_9_2b():
    """Semi-gradient temporal difference."""
    print("Test different alphas and steps for semi-gradient TD")
    n_groups = 20  # 20 aggregations
    ns = np.power(2, np.arange(0, 10))  # all possible steps
    alphas = np.arange(0, 1.1, 0.1)  # all possible alphas
    episodes = 10  # each run has 10 episodes
    runs = 100  # 100 independent runs

    # track RMS errors for each (step, alpha) combination
    errors = np.zeros((len(ns), len(alphas)))
    for _ in range(runs):
        for n_idx, n in enumerate(ns):
            for alpha_idx, alpha in enumerate(alphas):
                value_function = ValueFunction(N_STATES, n_groups)
                for _ in range(episodes):
                    semi_gradient_td(value_function, alpha, n)
                    state_values = np.asarray([value_function.value(i) for i in STATES])
                    errors[n_idx, alpha_idx] += np.sqrt(
                        np.mean((state_values - TRUE_VALUES) ** 2)
                    )

    errors /= episodes * runs  # average across episodes and runs

    for i in range(len(ns)):
        plt.plot(alphas, errors[i, :], label=f"n = {ns[i]}")
    plt.xlabel("alpha")
    plt.ylabel("RMS error")
    plt.ylim([0.25, 0.55])
    plt.legend()
    plt.show()


POLYNOMIAL_BASES = 0
FOURIER_BASES = 1


class BasesValueFunction:
    def __init__(self, btype: int, order: int, n_states: int):
        self.btype = btype  # type of bases
        self.order = order  # num of bases
        self.n_states = n_states
        if btype == POLYNOMIAL_BASES:
            self.bases = [lambda s, i=i: pow(s, i) for i in range(order + 1)]
        elif btype == FOURIER_BASES:
            self.bases = [
                lambda s, i=i: np.cos(i * np.pi * s) for i in range(order + 1)
            ]
        else:
            raise NotImplementedError
        self.weights = np.zeros(order + 1)

    def value(self, state):
        """Get state value."""
        _s = state / self.n_states  # map the state space into [0, 1]
        return self.weights.dot([func(_s) for func in self.bases])

    def update(self, delta, state):
        """Update weights."""
        _s = state / self.n_states  # map the state space into [0, 1]
        derivative_values = np.asarray([func(_s) for func in self.bases])
        self.weights += delta * derivative_values


def figure_9_5():
    """Polynomials and Fourier basis."""
    runs = 1
    episodes = 5000
    btypes = [POLYNOMIAL_BASES, FOURIER_BASES]
    alphas = [1e-4, 5e-5]
    orders = [5, 10, 20]  # of bases
    labels = [["polynomial basis"] * len(orders), ["fourier basis"] * len(orders)]

    # track errors for each episode
    errors = np.zeros((len(alphas), len(orders), episodes))
    for _ in range(runs):
        for btype_idx, (btype, alpha) in enumerate(zip(btypes, alphas)):
            for order_idx, order in enumerate(orders):
                value_function = BasesValueFunction(btype, order, N_STATES)
                for ep in range(episodes):
                    gradient_montecarlo(value_function, alpha)
                    state_values = np.asarray(
                        [value_function.value(state) for state in STATES]
                    )
                    errors[btype_idx, order_idx, ep] += np.sqrt(
                        np.mean((TRUE_VALUES - state_values) ** 2)
                    )

    errors /= runs  # average over runs

    for i in range(len(btypes)):
        for j in range(len(orders)):
            plt.plot(errors[i, j, :], label=f"{labels[i][j]} order = {orders[j]}")
    plt.xlabel("Episodes")
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()
