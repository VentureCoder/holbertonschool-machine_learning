#!/usr/bin/env python3
"""TD(lambda) algorithm with eligibility traces."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm.

    env is the environment instance
    V is a numpy.ndarray of shape (s,) holding the value estimate
    policy takes a state and returns the action to take
    lambtha is the eligibility trace factor
    episodes is the number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD error and accumulating eligibility trace for this state.
            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1

            # Update every state in proportion to its current trace,
            # then decay all traces by gamma * lambda.
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            if terminated or truncated:
                break
            state = next_state

    return V
