import heapq
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

WORLD_HEIGHT = 6
WORLD_WIDTH = 9

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3
ACTIONS = [NORTH, SOUTH, WEST, EAST]

START = [2, 0]
GOAL = [0, 8]

OBSTACLES = [[1, 2], [2, 2], [3, 2], [4, 5], [0, 7], [1, 7], [2, 7]]

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

    if new_state in OBSTACLES:
        new_state = [row, col]

    if new_state == GOAL:
        reward = 1.
    else:
        reward = 0.

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


class Agent:
    def __init__(self):
        self.model = defaultdict(defaultdict)

    def update(self, state, action, next_state, reward):
        """Update the model with previous experience."""
        raise NotImplementedError

    def sample(self):
        """Randomly sample from previous experience."""
        raise NotImplementedError


# Model for planning in Dyna-Q
class DynaQAgent(Agent):
    def __init__(self):
        super().__init__()

    def update(self, state, action, next_state, reward):
        self.model[tuple(state)][action] = [reward, list(next_state)]

    def sample(self):
        seen_states = list(self.model.keys())
        i = np.random.choice(range(len(seen_states)))
        state = seen_states[i]

        actions = list(self.model[state].keys())
        j = np.random.choice(range(len(actions)))
        action = actions[j]

        reward, next_state = self.model[state][action]
        return list(state), action, list(next_state), reward


# Model for planning in Dyna-Q+
class DynaQPlusAgent(Agent):
    def __init__(self, kappa=1e-4):
        super().__init__()
        self.kappa = kappa
        self.tau = 0  # time tracker

    def update(self, state, action, next_state, reward):
        self.tau += 1

        self.model[tuple(state)][action] = [reward, list(next_state), self.tau]

    def sample(self):
        seen_states = list(self.model.keys())
        i = np.random.choice(range(len(seen_states)))
        state = seen_states[i]

        actions = list(self.model[state].keys())
        j = np.random.choice(range(len(actions)))
        action = actions[j]

        reward, next_state, tau = self.model[state][action]
        reward += self.kappa * np.sqrt(self.tau - tau)  # adjust reward
        return list(state), action, list(next_state), reward


def dyna_q(
    q_values: np.ndarray,
    model: Agent,
    planning_step: int,
    max_t: float = float("inf"),
    gamma: float = 0.95,
    alpha: float = 0.1,
    epsilon: float = 0.1,
):
    t = 0
    state = START
    while t <= max_t and state != GOAL:
        action = policy_action(q_values, state, epsilon)
        next_state, reward = step(state, action)

        # Update
        target = np.max(q_values[next_state[0], next_state[1], :])
        q_values[state[0], state[1], action] += alpha * (
            reward + gamma * target - q_values[state[0], state[1], action]
        )
        model.update(state, action, next_state, reward)

        for _ in range(planning_step):
            state_, action_, next_state_, reward_ = model.sample()
            target_ = np.max(q_values[next_state_[0], next_state_[1], :])
            q_values[state_[0], state_[1], action_] += alpha * (
                reward_ + gamma * target_ - q_values[state_[0], state_[1], action_]
            )

        state = next_state
        t += 1
    
    return t


def prioritized_sweeping(
    q_values: np.ndarray,
    model: Agent,
    planning_step: int,
    max_t: float = float("inf"),
    gamma: float = 0.95,
    alpha: float = 0.1,
    epsilon: float = 0.1,
):
    t = 0
    state = START
    while t <= max_t and state != GOAL:
        action = policy_action(q_values, state, epsilon)
        next_state, reward = step(state, action)
        model.update(state, action, next_state, reward)
        p = np.abs(
            reward + gamma * np.max(q_values[next_state[0], next_state[1], :])
            - q_values[state[0], state[1], action]
        )
        if p > theta:
            pqueue.append((state))




def figure_8_2():
    runs = 10
    episodes = 50
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))

    for run in range(runs):
        print(f"run: {run}")
        for i, planning_step in enumerate(planning_steps):
            q_values = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))
            model = DynaQAgent()
            for j in range(episodes):
                steps[i, j] += dyna_q(q_values, model, planning_step)

    # averaging over runs
    steps /= runs

    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label=f"{planning_steps[i]} planning steps")
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.legend()
    plt.show()
