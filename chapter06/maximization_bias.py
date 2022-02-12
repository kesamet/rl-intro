from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

ACTIONS = {
    "A": ["left", "right"],
    "B": [f"left{i}" for i in range(10)],  # say 10 different actions
    "terminal": ["stay"],  # dummy action
}
START = "A"
GOAL = "terminal"


def init_q_values() -> dict:
    q_values = {}
    for s, actions in ACTIONS.items():
        q_values[s] = {a: 0. for a in actions}
    return q_values


def get_best_action(values: dict) -> str:
    max_val = np.max(list(values.values()))
    best_actions = [a for a, v_ in values.items() if v_ == max_val]
    return np.random.choice(best_actions)


def policy_action(
    q_values: dict,
    state: str,
    epsilon: float,
) -> np.ndarray:
    """Choose an action based on epsilon-greedy algorithm."""
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(ACTIONS[state])
    return get_best_action(q_values[state])


def step(state: str, action: str) -> Tuple[str, float]:
    """Get reward and next state given current state and action."""
    if state == "A":
        if action == "right":
            new_state = "terminal"
            reward = 0
        elif action == "left":
            new_state = "B"
            reward = 0
        else:
            assert False
    elif state == "B":
        new_state = "terminal"
        reward = np.random.randn() - 0.1
    else:
        assert False
    return new_state, reward


def q_learning(
    episodes: int,
    gamma: float = 1.,
    alpha: float = 0.5,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, list]:
    """Q-learning"""
    all_left_from_A = []

    q_values = init_q_values()
    for _ in range(episodes):
        state = START

        left_from_A = 0
        while state != GOAL:
            action = policy_action(q_values, state, epsilon)

            if state == "A" and action == "left":
                left_from_A += 1

            next_state, reward = step(state, action)

            # Update
            target = np.max(list(q_values[next_state].values()))
            q_values[state][action] += alpha * (
                reward + gamma * target - q_values[state][action]
            )

            state = next_state

        all_left_from_A.append(left_from_A)

    return q_values, all_left_from_A


def _sum_q(q_values1, q_values2):
    q_values = {}
    for s, actions in ACTIONS.items():
        q_values[s] = {}
        for a in actions:
            q_values[s][a] = q_values1[s][a] + q_values2[s][a]
    return q_values


def double_q_learning(
    episodes: int,
    gamma: float = 1.,
    alpha: float = 0.5,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Double Q-learning"""
    all_left_from_A = []

    q1 = init_q_values()
    q2 = init_q_values()
    for _ in range(episodes):
        state = START

        left_from_A = 0
        while state != GOAL:
            q_values_ = _sum_q(q1, q2)
            action = policy_action(q_values_, state, epsilon)

            if state == "A" and action == "left":
                left_from_A += 1

            next_state, reward = step(state, action)

            # Update
            if np.random.binomial(1, 0.5) == 1:
                active_q, target_q = q1, q2
            else:
                active_q, target_q = q2, q1

            best_action = get_best_action(active_q[next_state])
            target = target_q[next_state][best_action]
            active_q[state][action] += alpha * (
                reward + gamma * target - active_q[state][action]
            )

            state = next_state

        all_left_from_A.append(left_from_A)

    return q1, q2, all_left_from_A


def figure_6_5():
    runs = 1000
    episodes = 300

    # Multiple runs to smoothen curves
    r_qlearn = []
    r_dql = []
    for _ in range(runs):
        _, counts_qlearn = q_learning(episodes)
        r_qlearn.append(counts_qlearn)

        _, _, counts_dql = double_q_learning(episodes)
        r_dql.append(counts_dql)

    r_qlearn = np.array(r_qlearn).mean(axis=0)
    r_dql = np.array(r_dql).mean(axis=0)

    x = np.arange(1, episodes + 1)
    plt.plot(x, r_qlearn, label="Q-learning")
    plt.plot(x, r_dql, label="Double Q-learning")
    plt.plot(np.ones(episodes) * 0.05, label="Optimal")
    plt.xlabel("Episodes")
    plt.ylabel("% left actions from A")
    plt.legend()
    plt.show()
