from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_HEIGHT = 4
WORLD_WIDTH = 12

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3
ACTIONS = [NORTH, SOUTH, WEST, EAST]

START = [3, 0]
GOAL = [3, 11]


def step(state: list[int], action: np.ndarray) -> Tuple[List[int], float]:
    """Get reward and next state given current state and action."""
    row, col = state

    if action == NORTH:
        new_state = [max(row - 1, 0), col]
    elif action == SOUTH:
        new_state = [min(row + 1, WORLD_HEIGHT - 1), col]
    elif action == WEST:
        new_state = [row, max(col - 1, 0)]
    elif action == EAST:
        new_state = [row, min(col + 1, WORLD_WIDTH - 1)]
    reward = -1.0

    # Cliff
    if (action == SOUTH and row == 2 and 1 <= col <= 10) or (
        action == EAST and state == START
    ):
        new_state = START
        reward = -100.0

    return new_state, reward


def policy_action(
    q_values: np.ndarray,
    state: List[int],
    epsilon: float,
) -> np.ndarray:
    """Choose an action based on epsilon-greedy algorithm."""
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(ACTIONS)

    values_ = q_values[state[0], state[1], :]
    return np.random.choice(np.array(ACTIONS)[values_ == np.max(values_)])


def sarsa(
    episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.5,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, list]:
    """Sarsa."""
    all_rewards = []

    q_values = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
    for _ in range(episodes):
        state = START
        action = policy_action(q_values, state, epsilon)

        total_reward = 0
        while state != GOAL:
            next_state, reward = step(state, action)
            next_action = policy_action(q_values, next_state, epsilon)

            target = q_values[next_state[0], next_state[1], next_action]

            # Update
            q_values[state[0], state[1], action] += alpha * (
                reward + gamma * target - q_values[state[0], state[1], action]
            )
            state = next_state
            action = next_action
            total_reward += reward

        all_rewards.append(total_reward)
    return q_values, all_rewards


def expected_sarsa(
    episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.5,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, list]:
    """Expected Sarsa."""
    all_rewards = []

    q_values = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
    for _ in range(episodes):
        state = START
        action = policy_action(q_values, state, epsilon)

        total_reward = 0
        while state != GOAL:
            next_state, reward = step(state, action)
            next_action = policy_action(q_values, next_state, epsilon)

            v = q_values[next_state[0], next_state[1], :]
            n1 = np.sum(v == np.max(v))
            wgts = np.where(v == np.max(v), (1 - epsilon) / n1, 0) + epsilon / len(
                ACTIONS
            )
            target = wgts.dot(v)

            # Update
            q_values[state[0], state[1], action] += alpha * (
                reward + gamma * target - q_values[state[0], state[1], action]
            )
            state = next_state
            action = next_action
            total_reward += reward

        all_rewards.append(total_reward)
    return q_values, all_rewards


def q_learning(
    episodes: int,
    gamma: float = 1.0,
    alpha: float = 0.5,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, list]:
    """Q-learning"""
    all_rewards = []

    q_values = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
    for _ in range(episodes):
        state = START

        total_reward = 0
        while state != GOAL:
            action = policy_action(q_values, state, epsilon)
            next_state, reward = step(state, action)

            target = np.max(q_values[next_state[0], next_state[1], :])

            # Update
            q_values[state[0], state[1], action] += alpha * (
                reward + gamma * target - q_values[state[0], state[1], action]
            )
            state = next_state
            total_reward += reward

        all_rewards.append(total_reward)
    return q_values, all_rewards


def draw_optimal_policy(q_values, title=None):
    """Display optimal policy."""
    _, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = q_values.shape[:2]
    width, height = 1.0 / ncols, 1.0 / nrows

    optimal_policy = []
    for i in range(WORLD_HEIGHT):
        row = []
        for j in range(WORLD_WIDTH):
            if [i, j] == GOAL:
                val = "G"
            else:
                best_action = np.argmax(q_values[i, j, :])
                if best_action == NORTH:
                    val = "U"
                elif best_action == SOUTH:
                    val = "D"
                elif best_action == WEST:
                    val = "L"
                elif best_action == EAST:
                    val = "R"

            row.append(val)
            tb.add_cell(i, j, width, height, text=val, loc="center", facecolor="white")

        optimal_policy.append(row)

    ax.add_table(tb)
    if title:
        ax.set_title(title)
    plt.show()


def example_6_6():
    runs = 50
    episodes = 500

    # Multiple runs to smoothen curves
    r_sarsa = []
    r_qlearn = []
    for _ in range(runs):
        q_sarsa, all_rewards_sarsa = sarsa(episodes)
        r_sarsa.append(all_rewards_sarsa)

        q_qlearn, all_rewards_qlearn = q_learning(episodes)
        r_qlearn.append(all_rewards_qlearn)

    r_sarsa = np.array(r_sarsa).mean(axis=0)
    r_qlearn = np.array(r_qlearn).mean(axis=0)

    x = np.arange(1, episodes + 1)
    plt.plot(x, r_sarsa, label="Sarsa")
    plt.plot(x, r_qlearn, label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()

    draw_optimal_policy(q_sarsa, title="Sarsa optimal policy")
    draw_optimal_policy(q_qlearn, title="Q-learning optimal policy")
