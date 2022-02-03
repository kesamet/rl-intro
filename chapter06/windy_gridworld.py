from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_HEIGHT = 7  # world height
WORLD_WIDTH = 10  # world width

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3
NORTHEAST = 4
SOUTHEAST = 5
SOUTHWEST = 6
NORTHWEST = 7
STATIONARY = 8

REWARD = -1.0  # reward for each step

START = [3, 0]
GOAL = [3, 7]


class WindyGridWorld:
    def __init__(self, actions: list, randwind: bool = False, debug: bool = False):
        self.actions = actions
        self.randwind = randwind
        self.debug = debug

    def step(self, state: list[int], action: np.ndarray) -> Tuple[List[int], float]:
        """Get reward and next state given current state and action."""
        row, col = state
        wind = WIND[col]
        if self.randwind and wind > 0:
            wind += np.random.choice([-1, 0, 1])

        if action == NORTH:
            new_state = [max(row - 1 - wind, 0), col]
        elif action == SOUTH:
            new_state = [max(min(row + 1 - wind, WORLD_HEIGHT - 1), 0), col]
        elif action == WEST:
            new_state = [max(row - wind, 0), max(col - 1, 0)]
        elif action == EAST:
            new_state = [max(row - wind, 0), min(col + 1, WORLD_WIDTH - 1)]
        elif action == NORTHEAST:
            new_state = [max(row - 1 - wind, 0), min(col + 1, WORLD_WIDTH - 1)]
        elif action == SOUTHEAST:
            new_state = [
                max(min(row + 1 - wind, WORLD_HEIGHT - 1), 0),
                min(col + 1, WORLD_WIDTH - 1),
            ]
        elif action == SOUTHWEST:
            new_state = [max(min(row + 1 - wind, WORLD_HEIGHT - 1), 0), max(col - 1, 0)]
        elif action == NORTHWEST:
            new_state = [max(row - 1 - wind, 0), max(col - 1, 0)]
        elif action == STATIONARY:
            new_state = [max(row - wind, 0), col]
        else:
            assert False
        return new_state, REWARD

    def policy_action(
        self,
        q_values: np.ndarray,
        state: List[int],
        epsilon: float,
    ) -> np.ndarray:
        """Choose an action based on epsilon-greedy algorithm."""
        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(range(len(self.actions)))

        values_ = q_values[state[0], state[1], :]
        return np.random.choice(
            [i for i, v_ in enumerate(values_) if v_ == np.max(values_)])

    def sarsa(
        self,
        episodes: int,
        gamma: float = 1.,
        alpha: float = 0.5,
        epsilon: float = 0.1,
    ) -> Tuple[np.ndarray, list]:
        """Sarsa."""
        steps = []

        q_values = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(self.actions)))
        for _ in range(episodes):
            state = START
            action = self.policy_action(q_values, state, epsilon)

            if self.debug:
                print(state)
            iters = 0
            while state != GOAL:
                next_state, reward = self.step(state, action)
                if self.debug:
                    print(f"+ {action} ==> {next_state}")
                next_action = self.policy_action(q_values, next_state, epsilon)
                q_values[state[0], state[1], action] += alpha * (
                    reward + gamma * q_values[next_state[0], next_state[1], next_action]
                    - q_values[state[0], state[1], action]
                )
                state = next_state
                action = next_action
                iters += 1
            
            steps.append(iters)
        return q_values, steps


def draw_optimal_policy(q_values, title=None):
    """Draw optimal policy."""
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
            tb.add_cell(
                i, j, width, height, text=val, loc="center", facecolor="white")
    
        optimal_policy.append(row)

    for j in range(ncols):
        tb.add_cell(
            nrows, j, width, height/2, text=WIND[j],
            loc="center", edgecolor=None, facecolor="none",
        )

    ax.add_table(tb)
    if title:
        ax.set_title(title)
    plt.show()


def figure_6_5():
    world = WindyGridWorld([NORTH, SOUTH, WEST, EAST])
    q_values, steps = world.sarsa(160)

    plt.plot(np.add.accumulate(steps), np.arange(1, len(steps) + 1))
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.show()

    draw_optimal_policy(q_values, title="Sarsa optimal policy")


def exercise_6_9():
    world = WindyGridWorld([
        NORTH, SOUTH, WEST, EAST, NORTHEAST, SOUTHEAST, SOUTHWEST, NORTHWEST])
    _, steps = world.sarsa(160)

    plt.plot(np.add.accumulate(steps), np.arange(1, len(steps) + 1), label="Without stationary")
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")

    world = WindyGridWorld([
        NORTH, SOUTH, WEST, EAST, NORTHEAST, SOUTHEAST, SOUTHWEST, NORTHWEST, STATIONARY])
    _, steps = world.sarsa(160)

    plt.plot(np.add.accumulate(steps), np.arange(1, len(steps) + 1), label="With stationary")
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.legend()
    plt.show()


def exercise_6_10():
    world = WindyGridWorld([
        NORTH, SOUTH, WEST, EAST, NORTHEAST, SOUTHEAST, SOUTHWEST, NORTHWEST, STATIONARY],
        randwind=True,
    )
    _, steps = world.sarsa(160)

    plt.plot(np.add.accumulate(steps), np.arange(1, len(steps) + 1))
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.show()
