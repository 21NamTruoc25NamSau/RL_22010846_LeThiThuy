import gymnasium as gym
import numpy as np
from collections import defaultdict

def generate_episode(env, policy=None, seed=None):
    """Sinh 1 episode từ môi trường theo policy cho trước."""
    state, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    episode = []
    while not (terminated or truncated):
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode

def compute_returns(rewards, gamma=1.0):
    """Tính discounted return G_t từ danh sách rewards từ cuối episode trở về đầu."""
    returns = [0.0] * len(rewards)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    return 0 if player_sum >= 20 else 1


def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        visited_states = set()
        for idx, step in enumerate(ep):
            state = step[0]
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += g_list[idx]
                returns_count[state] += 1
    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}
    return V, returns_count


def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        visited_states = set()
        for idx, step in enumerate(ep):
            state = step[0]
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += g_list[idx]
                returns_count[state] += 1
    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}
    return V, returns_count


def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        visited_states = set()
        for idx, step in enumerate(ep):
            state = step[0]
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += g_list[idx]
                returns_count[state] += 1
    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}
    return V, returns_count


def every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        for idx, step in enumerate(ep):
            state = step[0]
            returns_sum[state] += g_list[idx]
            returns_count[state] += 1
    V = {state: returns_sum[state] / returns_count[state] for state in returns_sum}
    return V, returns_count


def mc_action_value_prediction(env, policy, n_episodes, gamma=1.0):
    from collections import defaultdict
    import numpy as np
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        visited_sa = set()
        for idx, step in enumerate(ep):
            st, action, _ = step
            sa = (st, action)
            if sa not in visited_sa:
                visited_sa.add(sa)
                returns_sum[sa] += g_list[idx]
                returns_count[sa] += 1
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for (st, action), total_r in returns_sum.items():
        Q[st][action] = total_r / returns_count[(st, action)]
    return Q


def mc_action_value_prediction(env, policy, n_episodes, gamma=1.0):
    from collections import defaultdict
    import numpy as np
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        visited_sa = set()
        for idx, step in enumerate(ep):
            st, action, _ = step
            sa = (st, action)
            if sa not in visited_sa:
                visited_sa.add(sa)
                returns_sum[sa] += g_list[idx]
                returns_count[sa] += 1
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for (st, action), total_r in returns_sum.items():
        Q[st][action] = total_r / returns_count[(st, action)]
    return Q


def mc_action_value_prediction(env, policy, n_episodes, gamma=1.0):
    from collections import defaultdict
    import numpy as np
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=ep_idx)
        rewards = [step[2] for step in ep]
        g_list = compute_returns(rewards, gamma=gamma)
        visited_sa = set()
        for idx, step in enumerate(ep):
            st, action, _ = step
            sa = (st, action)
            if sa not in visited_sa:
                visited_sa.add(sa)
                returns_sum[sa] += g_list[idx]
                returns_count[sa] += 1
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for (st, action), total_r in returns_sum.items():
        Q[st][action] = total_r / returns_count[(st, action)]
    return Q


def greedy_action(Q, state, n_actions):
    import numpy as np
    if state not in Q:
        return 0
    return int(np.argmax(Q[state]))


def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    import numpy as np
    if rng.random() < epsilon:
        return rng.integers(0, n_actions)
    else:
        if state not in Q:
            return rng.integers(0, n_actions)
        return int(np.argmax(Q[state]))


def update_incremental_mean(Q, N, state, action, G):
    N[state][action] += 1
    Q[state][action] += (G - Q[state][action]) / N[state][action]


def generate_episode_control(env, Q, epsilon, rng):
    episode = []
    state, _ = env.reset()
    done = False
    n_actions = env.action_space.n
    while not done:
        action = epsilon_greedy_action(Q, state, n_actions, epsilon, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state
    return episode

def on_policy_mc_control(env, n_episodes, gamma=1.0, epsilon=0.1, seed=42):
    import numpy as np
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))
    episode_rewards = []
    for ep_idx in range(n_episodes):
        ep = generate_episode_control(env, Q, epsilon, rng)
        rewards = [step[2] for step in ep]
        episode_rewards.append(sum(rewards))
        g_list = compute_returns(rewards, gamma=gamma)
        visited_sa = set()
        for idx, step in enumerate(ep):
            st, act, _ = step
            sa = (st, act)
            if sa not in visited_sa:
                visited_sa.add(sa)
                update_incremental_mean(Q, N, st, act, g_list[idx])
    def policy(state):
        return greedy_action(Q, state, n_actions)
    return Q, policy, episode_rewards


def generate_episode_control(env, Q, epsilon, rng):
    episode = []
    state, _ = env.reset()
    done = False
    n_actions = env.action_space.n
    while not done:
        action = epsilon_greedy_action(Q, state, n_actions, epsilon, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append((state, action, reward))
        state = next_state
    return episode

def on_policy_mc_control(env, n_episodes, gamma=1.0, epsilon=0.1, seed=42):
    import numpy as np
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))
    episode_rewards = []
    for ep_idx in range(n_episodes):
        ep = generate_episode_control(env, Q, epsilon, rng)
        rewards = [step[2] for step in ep]
        episode_rewards.append(sum(rewards))
        g_list = compute_returns(rewards, gamma=gamma)
        visited_sa = set()
        for idx, step in enumerate(ep):
            st, act, _ = step
            sa = (st, act)
            if sa not in visited_sa:
                visited_sa.add(sa)
                update_incremental_mean(Q, N, st, act, g_list[idx])
    def policy(state):
        return greedy_action(Q, state, n_actions)
    return Q, policy, episode_rewards


def evaluate_policy(env, policy, n_episodes=10000, seed=123):
    import numpy as np
    wins, losses, draws = 0, 0, 0
    total_rewards = []
    for ep_idx in range(n_episodes):
        ep = generate_episode(env, policy=policy, seed=seed + ep_idx)
        final_r = ep[-1][2]
        total_rewards.append(final_r)
        if final_r == 1.0:
            wins += 1
        elif final_r == -1.0:
            losses += 1
        else:
            draws += 1
    return {
        'win_rate': wins / n_episodes,
        'loss_rate': losses / n_episodes,
        'draw_rate': draws / n_episodes,
        'mean_reward': np.mean(total_rewards)
    }
