#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]

# left, up, right, down
ACTIONS = [
    np.array([0, -1]),
    np.array([-1, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
]
ACTIONS_FIGS = [ "←", "↑", "→", "↓"]

ACTION_PROB = 0.25
GAMMA = 0.9


def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


def draw_table(arr):
    _, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = arr.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(arr):
        text = f"{val:.1f}"
        # add state labels
        if [i, j] == A_POS:
            text += " (A)"
        if [i, j] == A_PRIME_POS:
            text += " (A')"
        if [i, j] == B_POS:
            text += " (B)"
        if [i, j] == B_PRIME_POS:
            text += " (B')"
        
        tb.add_cell(
            i, j, width, height, text=text, loc="center", facecolor="white")
        
    # Row and column labels
    for i in range(nrows):
        tb.add_cell(
            i, -1, width, height, text=i+1, loc="right", 
            edgecolor="none", facecolor="none")
    for j in range(ncols):
        tb.add_cell(
            -1, j, width, height/2, text=j+1, loc="center",
            edgecolor="none", facecolor="none")

    ax.add_table(tb)
    plt.show()


def draw_policy(optimal_values):
    _, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), _ in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0], next_state[1]])

        best_actions = np.where(next_vals == np.max(next_vals))[0]
        text = "".join([ACTIONS_FIGS[a] for a in best_actions])

        # add state labels
        if [i, j] == A_POS:
            text += " (A)"
        if [i, j] == A_PRIME_POS:
            text += " (A')"
        if [i, j] == B_POS:
            text += " (B)"
        if [i, j] == B_PRIME_POS:
            text += " (B')"
        
        tb.add_cell(
            i, j, width, height, text=text, loc="center", facecolor="white")

    # Row and column labels
    for i in range(nrows):
        tb.add_cell(
            i, -1, width, height, text=i+1, loc="right", 
            edgecolor="none", facecolor="none")
    for j in range(ncols):
        tb.add_cell(
            -1, j, width, height/2, text=j+1, loc="center",
            edgecolor="none", facecolor="none")
    ax.add_table(tb)
    plt.show()


def figure_3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + GAMMA * value[next_i, next_j])

        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_table(new_value)
            break

        value = new_value


def figure_3_2_exact():
    """Using linear system of equations to find the exact solution."""
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))

                A[index_s, index_s_] += ACTION_PROB * GAMMA
                b[index_s] -= ACTION_PROB * r

    x = np.linalg.solve(A, b)
    draw_table(x.reshape(WORLD_SIZE, WORLD_SIZE))


def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + GAMMA * value[next_i, next_j])
                new_value[i, j] = np.max(values)

        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_table(new_value)
            draw_policy(new_value)
            break

        value = new_value


if __name__ == "__main__":
    figure_3_2_exact()
    figure_3_2()
    figure_3_5()
