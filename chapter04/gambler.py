import numpy as np
import matplotlib.pyplot as plt

# goal
GOAL = 100

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)


def value_iteration(head_prob: float = 0.4, theta: float = 1e-8) -> tuple:
    """4.4 Value iteration."""
    state_values = np.zeros(len(STATES))
    state_values[GOAL] = 1.0

    while True:
        old_state_values = state_values.copy()
        for state in STATES[1:GOAL]:
            rets = []
            for action in range(1, min(state, GOAL - state) + 1):
                if state + action < GOAL:
                    rets.append(
                        head_prob * state_values[state + action]
                        + (1 - head_prob) * state_values[state - action]
                    )
                else:
                    rets.append(
                        head_prob + (1 - head_prob) * state_values[state - action]
                    )
            state_values[state] = max(rets)

        if abs(state_values - old_state_values).max() < theta:
            break

    policy = np.zeros_like(STATES)
    for state in STATES[1:GOAL]:
        actions = np.arange(1, min(state, GOAL - state) + 1)
        rets = []
        for action in actions:
            if state + action < GOAL:
                rets.append(
                    head_prob * state_values[state + action]
                    + (1 - head_prob) * state_values[state - action]
                )
            else:
                rets.append(head_prob + (1 - head_prob) * state_values[state - action])

        policy[state] = actions[np.argmax(rets)]

    return state_values, policy


def figure_4_3():
    state_values, policy = value_iteration()

    plt.plot(state_values)
    plt.xlabel("Capital")
    plt.ylabel("Value estimates")
    plt.legend(loc="best")
    plt.show()

    plt.bar(STATES, policy)
    plt.xlabel("Capital")
    plt.ylabel("Final policy (stake)")
    plt.show()


def train(ph=0.4, Theta=1e-6):
    """Alternative"""
    print(f"ph = {ph:.4f}")
    V = [np.random.random() * 1000 for _ in range(100)]
    V[0] = 0
    pi = [0] * 100
    counter = 1
    while True:
        Delta = 0
        for s in range(1, 100):  # for each state
            old_v = V[s]
            v = [0] * 51
            for a in range(1, min(s, 100 - s) + 1):
                v[a] = 0
                if a + s < 100:
                    v[a] += ph * (0 + V[s + a])
                    v[a] += (1 - ph) * (0 + V[s - a])
                elif a + s == 100:
                    v[a] += ph
                    v[a] += (1 - ph) * (0 + V[s - a])
            op_a = np.argmax(v)
            pi[s] = op_a
            V[s] = v[op_a]
            Delta = max(Delta, abs(old_v - V[s]))
        counter += 1
        if counter % 1000 == 0:
            print(f"  train loop {counter}")
            print(f"  Delta = {Delta}")
        if Delta < Theta:
            print(f"  train loop {counter}")
            print(f"  Delta = {Delta}\n")
            break
    return [V[1:100], pi[1:100]]
