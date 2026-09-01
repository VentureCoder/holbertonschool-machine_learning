#!/usr/bin/env python3
"""SARSA(lambda) algorithm with eligibility traces."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Performs the SARSA(lambda) algorithm."""
    def epsilon_greedy(state, eps):
        if np.random.uniform() > eps:
            return np.argmax(Q[state])
        return np.random.randint(0, Q.shape[1])

    max_epsilon = epsilon
    for episode in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(state, epsilon)
        eligibility = np.zeros_like(Q)
        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(next_state, epsilon)
            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            if terminated or truncated:
                break
            state, action = next_state, next_action
        epsilon = (min_epsilon + (max_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * episode))
    return Q
