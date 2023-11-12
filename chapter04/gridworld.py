from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = {
    "left": {
        "vec": np.array([0, -1]),
        "prob": 0.25,
    },
    "up": {
        "vec": np.array([-1, 0]),
        "prob": 0.25,
    },
    "right": {
        "vec": np.array([0, 1]),
        "prob": 0.25,
    },
    "down": {
        "vec": np.array([1, 0]),
        "prob": 0.25,
    },
}


def is_terminal(state: List[int]) -> bool:
    return (state[0] == 0 and state[1] == 0) or (
        state[0] == WORLD_SIZE - 1 and state[1] == WORLD_SIZE - 1
    )


def out_of_grid(state: List[int]) -> bool:
    return (
        state[0] < 0 or state[0] >= WORLD_SIZE or state[1] < 0 or state[1] >= WORLD_SIZE
    )


def step(state: List[int], action) -> Tuple[List[int], float]:
    if is_terminal(state):
        return state, 0  # state, reward

    next_state = (np.array(state) + action).tolist()
    if out_of_grid(next_state):
        next_state = state

    return next_state, -1  # state, reward


def compute_state_values(
    inplace: bool = True, gamma: float = 1.0, theta: float = 1e-4
) -> np.ndarray:
    """4.1 Iterative policy evaluation (DP)."""
    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iters = 0
    while True:
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                _v = 0
                for a in ACTIONS.values():
                    (next_i, next_j), reward = step([i, j], a["vec"])
                    if inplace:
                        # Asynchronous/inplace
                        _v += a["prob"] * (
                            reward + gamma * state_values[next_i, next_j]
                        )
                    else:
                        # Synchronous
                        _v += a["prob"] * (
                            reward + gamma * old_state_values[next_i, next_j]
                        )
                state_values[i, j] = _v

        if np.abs(state_values - old_state_values).max() < theta:
            break

        iters += 1

    print(f"  {iters} iterations")
    return state_values


def draw_table(arr: np.ndarray) -> None:
    _, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = arr.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(arr):
        tb.add_cell(i, j, width, height, text=val, loc="center", facecolor="white")

    # Row and column labels...
    for i in range(len(arr)):
        tb.add_cell(
            i,
            -1,
            width,
            height,
            text=i + 1,
            loc="right",
            edgecolor="none",
            facecolor="none",
        )
        tb.add_cell(
            -1,
            i,
            width,
            height / 2,
            text=i + 1,
            loc="center",
            edgecolor="none",
            facecolor="none",
        )
    ax.add_table(tb)
    plt.show()


def figure_4_1():
    print("In-place:")
    state_values = compute_state_values(inplace=True)
    draw_table(np.round(state_values, decimals=2))

    print("Synchronous:")
    state_values = compute_state_values(inplace=False)
    draw_table(np.round(state_values, decimals=2))
