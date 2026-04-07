"""
Bonus — PPO (Proximal Policy Optimization) for Pyrace-v3
Replaces hand-coded DQN with stable-baselines3's PPO.
PPO is a policy gradient method — more advanced than DQN because:
  - It learns a policy directly (not just Q-values)
  - Uses clipped surrogate objective for stable updates
  - On-policy: learns from fresh experience each update

Run training:   python Pyrace_RL_Bonus.py train
Run playback:   python Pyrace_RL_Bonus.py play
"""
import os
import sys
import gymnasium as gym
import gym_race
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

VERSION_NAME = "PPO_bonus_v3"
os.makedirs(f"models_{VERSION_NAME}", exist_ok=True)


def train():
    env = gym.make("Pyrace-v3")
    env = Monitor(env, f"models_{VERSION_NAME}/monitor.csv")
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=f"models_{VERSION_NAME}/",
        name_prefix="ppo_checkpoint"
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    print("Starting PPO training...")
    model.learn(total_timesteps=500_000, callback=checkpoint_cb)
    model.save(f"models_{VERSION_NAME}/ppo_pyrace_final")
    print("Training complete! Model saved.")
    env.close()


def load_and_play(model_path):
    os.environ.pop("SDL_VIDEODRIVER", None)
    import pygame
    pygame.init()
    play_env = gym.make("Pyrace-v3").unwrapped
    play_env.pyrace.enable_display()
    play_env.set_view(True)
    play_env.pyrace.mode = 2

    loaded_model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    for episode in range(10):
        obs, _ = play_env.reset()
        total_reward = 0
        for t in range(2000):
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, _, info = play_env.step(int(action))
            total_reward += reward
            play_env.set_msgs([
                "PPO BONUS",
                f"Episode: {episode}",
                f"Time steps: {t}",
                f'check: {info["check"]}',
                f'dist: {info["dist"]}',
                f'crash: {info["crash"]}',
                f"Reward: {total_reward:.0f}",
            ])
            play_env.render()
            if done:
                print(f"Episode {episode} finished | reward: {total_reward:.0f} | steps: {t}")
                break

    play_env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "play"
    if mode == "train":
        train()
    else:
        load_and_play(f"models_{VERSION_NAME}/ppo_pyrace_final")