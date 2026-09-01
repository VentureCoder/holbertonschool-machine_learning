#!/usr/bin/env python3
"""Monte Carlo value-estimation algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm.

    env is the environment instance
    V is a numpy.ndarray of shape (s,) holding the value estimate
    policy takes a state and returns the action to take
    episodes is the number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()[0]
        states = [state]
        rewards = []

        for _ in range(max_steps):
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            if terminated or truncated:
                break

        # Walk the episode backward, accumulate the return, update each
        # visited state toward it (every-visit).
        G = 0
        for st, reward in zip(states[:-1][::-1], rewards[::-1]):
            G = gamma * G + reward
            V[st] = V[st] + alpha * (G - V[st])

    return V
