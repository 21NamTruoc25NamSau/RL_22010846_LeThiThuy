"""
mdp_utils.py
Các hàm dùng chung cho Dynamic Programming trên FrozenLake-v1.
"""
import numpy as np

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
ACTION_SYMBOLS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def q_from_v(env, V, state, action, gamma):
    P = env.unwrapped.P
    q = 0.0
    for probability, next_state, reward, terminated in P[state][action]:
        q += probability * (reward + gamma * V[next_state] * (not terminated))
    return q


def action_values(env, V, state, gamma):
    n_actions = env.action_space.n
    q_values = np.zeros(n_actions)
    for a in range(n_actions):
        q_values[a] = q_from_v(env, V, state, a, gamma)
    return q_values


def policy_evaluation_sweep(env, policy, V, gamma):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    new_V = np.zeros(n_states)
    for s in range(n_states):
        v = 0.0
        for a in range(n_actions):
            pi_a_s = policy[s][a]
            if pi_a_s == 0:
                continue
            v += pi_a_s * q_from_v(env, V, s, a, gamma)
        new_V[s] = v
    return new_V


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n)
    for i in range(1, max_iterations + 1):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)
        delta = np.max(np.abs(new_V - V))
        V = new_V
        if delta < theta:
            return V, i
    return V, max_iterations


def policy_evaluation_with_history(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n)
    deltas = []
    for i in range(1, max_iterations + 1):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            break
    return V, i, deltas


def greedy_policy_from_value(env, V, gamma=0.99):
    n_states = env.observation_space.n
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        policy[s] = np.argmax(action_values(env, V, s, gamma))
    return policy


def greedy_policy_matrix_from_value(env, V, gamma=0.99):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        best_a = np.argmax(action_values(env, V, s, gamma))
        policy[s, best_a] = 1.0
    return policy


def print_policy(policy):
    for s, a in enumerate(policy):
        print(f"State {s} -> Action {a} ({ACTION_NAMES.get(int(a), a)})")


def print_frozenlake_policy(env, policy, grid_shape=(4, 4)):
    desc = env.unwrapped.desc.astype(str)
    n_rows, n_cols = grid_shape
    lines = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            s = r * n_cols + c
            cell = desc[r][c]
            row.append(cell if cell in ("H", "G") else ACTION_SYMBOLS[int(policy[s])])
        lines.append(" ".join(row))
    print("\n".join(lines))
    return "\n".join(lines)


def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    policy = np.ones((n_states, n_actions)) / n_actions
    for i in range(1, max_iterations + 1):
        V, _ = policy_evaluation(env, policy, gamma, theta)
        new_policy = greedy_policy_matrix_from_value(env, V, gamma)
        policy_stable = np.array_equal(np.argmax(new_policy, axis=1), np.argmax(policy, axis=1))
        policy = new_policy
        if policy_stable:
            return np.argmax(policy, axis=1), V, i
    return np.argmax(policy, axis=1), V, max_iterations


def value_iteration_sweep(env, V, gamma):
    n_states = env.observation_space.n
    new_V = np.zeros(n_states)
    for s in range(n_states):
        new_V[s] = np.max(action_values(env, V, s, gamma))
    return new_V


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n)
    deltas = []
    for i in range(1, max_iterations + 1):
        new_V = value_iteration_sweep(env, V, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            return V, i, deltas
    return V, max_iterations, deltas


def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    rng = np.random.default_rng(seed)
    successes, total_rewards, lengths = 0, [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        terminated = truncated = False
        ep_reward = ep_length = 0
        while not (terminated or truncated):
            action = int(policy[obs])
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
        total_rewards.append(ep_reward)
        lengths.append(ep_length)
        if ep_reward > 0:
            successes += 1
    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
    }
