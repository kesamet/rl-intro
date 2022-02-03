# TODO
import time
import multiprocessing as mp
from functools import partial
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

PARALLEL_PROCESSES = 4
MAX_CARS = 20  # maximum # of cars in each location
MAX_MOVE = 5  # maximum # of cars to move during night
MOVE_COST = 2  # cost of moving a car
RENTAL_CREDIT = 10  # credit earned by a car
AVG_REQUESTS_LOC1 = 3  # average rental requests in first location
AVG_REQUESTS_LOC2 = 4  # average rental requests in second location
AVG_RETURNS_LOC1 = 3  # average number of cars returned in first location
AVG_RETURNS_LOC2 = 2  # average number of cars returned in second location
ADD_PARK_COST = 4


class PolicyIteration:
    def __init__(self, delta=1e-2, gamma=0.9, poisson_upper_bound=11, solve_4_7=False):
        self.solve_4_7 = solve_4_7
        self.delta = delta
        self.gamma = gamma
        # An upper bound for poisson distribution
        # If n is greater than this value, then the probability of getting n is truncated to 0
        self.poisson_upper_bound = poisson_upper_bound
        self.poisson = {
            lam: {
                n: poisson.pmf(n, lam) for n in range(poisson_upper_bound)
            } for lam in set([AVG_REQUESTS_LOC1, AVG_REQUESTS_LOC2, AVG_RETURNS_LOC1, AVG_RETURNS_LOC2])
        }

    def solve(self):    
        state_values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        policy = np.zeros_like(state_values, dtype=int)
        actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)

        t0 = time.time()
        iters = 0
        while True:
            start_time = time.time()
            print(f"Iteration {iters}: PE")
            state_values = self.policy_evaluation(state_values, policy)
            print(f"  Elapsed time: {time.time() - start_time:.2f} secs")
            
            start_time = time.time()
            print(f"Iteration {iters}: PI")
            policy, policy_changes = self.policy_improvement(state_values, policy, actions)
            print(f"  Policy changed in {policy_changes} states")
            print(f"  Elapsed time: {time.time() - start_time:.2f} secs")
        
            iters += 1
            if policy_changes == 0:
                break

        print(f"\nOptimal policy is reached after {iters} iterations in {time.time() - t0:.2f} secs")
        self.plot(policy)
        return policy

    def plot(self, policy):
        plt.figure()
        plt.table(cellText=np.flipud(policy), loc=(0, 0), cellLoc="center")
        plt.xlim(0, MAX_CARS + 1)
        plt.ylim(0, MAX_CARS + 1)
        plt.show()

    def bellman(self, action: int, state_values: np.ndarray, state: np.ndarray) -> float:
        """O(n^4) computation for all possible requests and returns."""
        # cost for moving cars
        r = -MOVE_COST * abs(action)
        if self.solve_4_7 and action > 0:
            # Free one shuttle to the second location
            r += MOVE_COST

        for req1 in range(self.poisson_upper_bound):
            for req2 in range(self.poisson_upper_bound):
                # moving cars
                num_loc1 = min(state[0] - action, MAX_CARS)
                num_loc2 = min(state[1] + action, MAX_CARS)

                # valid rental requests should be less than actual # of cars
                rentals_loc1 = min(req1, num_loc1)
                rentals_loc2 = min(req2, num_loc2)

                # get credits for renting
                reward = (rentals_loc1 + rentals_loc2) * RENTAL_CREDIT

                if self.solve_4_7:
                    if num_loc1 > 10:
                        reward += ADD_PARK_COST
                    if num_loc2 > 10:
                        reward += ADD_PARK_COST

                num_loc1 -= rentals_loc1
                num_loc2 -= rentals_loc2

                # probability for current combination of rental requests
                prob = self.poisson[AVG_REQUESTS_LOC1][req1] * self.poisson[AVG_REQUESTS_LOC2][req2]
                for ret1 in range(self.poisson_upper_bound):
                    for ret2 in range(self.poisson_upper_bound):
                        _num_loc1 = min(num_loc1 + ret1, MAX_CARS)
                        _num_loc2 = min(num_loc2 + ret2, MAX_CARS)
                        _prob = prob * (
                            self.poisson[AVG_RETURNS_LOC1][ret1] * self.poisson[AVG_RETURNS_LOC2][ret2]
                        )
                        r += _prob * (reward + self.gamma * state_values[_num_loc1, _num_loc2])
        return r

    def expected_return_pe(self, policy: np.ndarray, state_values: np.ndarray, state: np.ndarray) -> tuple:
        action = policy[state[0], state[1]]
        expected_return = self.bellman(action, state_values, state)
        return expected_return, state[0], state[1]

    def policy_evaluation(self, state_values: np.ndarray, policy: np.ndarray) -> np.ndarray:
        """4.3 Policy Evaluation."""
        while True:
            new_state_values = state_values.copy()
            with mp.Pool(processes=PARALLEL_PROCESSES) as p:
                all_states = ((i, j) for i, j in product(np.arange(MAX_CARS + 1), np.arange(MAX_CARS + 1)))
                cook = partial(self.expected_return_pe, policy, state_values)
                results = p.map(cook, all_states)

            for v, i, j in results:
                new_state_values[i, j] = v

            d = np.abs(new_state_values - state_values).sum()
            print(f"  - delta = {d}")
            state_values = new_state_values
            if d < self.delta:
                print("  Values have converged!")
                break
                
        return state_values

    def expected_return_pi(self, action: int, state_values: np.ndarray, state: np.ndarray) -> tuple:
        if 0 <= action <= state[0] or -state[1] <= action <= 0:
            expected_return = self.bellman(action, state_values, state)
            return expected_return, state[0], state[1], action
        return -float("inf"), state[0], state[1], action

    def policy_improvement(self, state_values: np.ndarray, policy: np.ndarray, actions: np.ndarray) -> tuple:
        """4.3 Policy Improvement."""
        action_idxs = {el: i for i, el in enumerate(actions)}
        new_policy = policy.copy()
        expected_action_returns = np.zeros((MAX_CARS + 1, MAX_CARS + 1, np.size(actions)))
        for action in actions:
            with mp.Pool(processes=PARALLEL_PROCESSES) as p:
                all_states = ((i, j) for i, j in product(np.arange(MAX_CARS + 1), np.arange(MAX_CARS + 1)))
                cook = partial(self.expected_return_pi, action, state_values)
                results = p.map(cook, all_states)

            for v, i, j, a in results:
                expected_action_returns[i, j, action_idxs[a]] = v

        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                new_policy[i, j] = actions[np.argmax(expected_action_returns[i, j])]

        policy_changes = (new_policy != policy).sum()
        return new_policy, policy_changes


if __name__ == "__main__":
    print("Figure 4.2")
    solver = PolicyIteration(solve_4_7=False)
    _ = solver.solve()

    print("Exercise 4.7")
    solver = PolicyIteration(solve_4_7=True)
    _ = solver.solve()
