"""Bài 4: Một episode ngẫu nhiên"""
import gymnasium as gym

def main():
    env = gym.make("Blackjack-v1")
    state, _ = env.reset(seed=42)
    terminated = False
    truncated = False
    step = 0
    print(f"Trạng thái bắt đầu (state): {state}\n")

    while not (terminated or truncated):
        step += 1
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        print(f"--- Bước {step} ---")
        print(f"  state      : {state}")
        print(f"  action     : {action}")
        print(f"  reward     : {reward}")
        print(f"  next_state : {next_state}")
        print(f"  terminated : {terminated}")
        print(f"  truncated  : {truncated}\n")

        state = next_state

    env.close()

if __name__ == "__main__":
    main()
