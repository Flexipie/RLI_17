"""
DQN for Pyrace-v3 — Part 2: improvements over the baseline (Pyrace-v1 / Part 1).

== Improvements implemented ==

1. Continuous observations
   Compared to the first version, the distance observations are more precise.
   Instead of collapsing every 20 pixels into 1 value, we return the exact distance.

2. BRAKE action
   First DQN only had acceleration, and turning movements.
   The movement was slowed down only by friction.
   For the improved version, BRAKE has been added which gives the agent direct control over velocity.

== Observations ==

We noticed the main limitation of this approach: catastrophic forgetting.
After training the DQN for a while, the performance started to degrade rapidly.
The sweet spot for the model was around 3500 episodes where it could comfortably complete laps.
The main issue is that the model learned that going slower is the safe option.
It completes laps but does not utilise speed to its fullest potential.
A fix to this problem could be to increase the importance of speed in the reward function.

Update after increasing the influence of speed in the reward function: model is now willing to go faster and took less time to complete laps consistently.

"""
from __future__ import annotations

import math
import os
import random
from collections import deque

import gymnasium as gym
import gym_race
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

VERSION_NAME = "DQN_v02"

REPORT_EPISODES    = 500
REPLAY_CAPACITY    = 100_000
BATCH_SIZE         = 128
WARMUP_STEPS       = 512
TRAIN_UPDATES_PER_STEP = 1
HIDDEN_DIM         = 256
ADAM_LR            = 1e-3
# observations are now continuous 0–200 px; scale rewards by the same order of magnitude
REWARD_SCALE       = 10_000.0
GRAD_CLIP_NORM     = 10.0
TARGET_UPDATE_EVERY = 500
LOG_EVERY_EPISODES  = 50
MAX_EXPLORE_RATE    = 1.0
EPSILON_DECAY_STEPS = 30_000
# lightweight extra shaping on top of the env's built-in dense reward
PROGRESS_REWARD_SCALE = 0.01
SURVIVAL_REWARD    = 0.0   # env already gives speed reward, so no extra survival bonus needed
N_ENVS             = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

env         = None
policy_net  = None
target_net  = None
optimizer   = None
replay_buffer = None
obs_high: np.ndarray | None = None


class QNetwork(nn.Module):
    def __init__(self, n_obs: int, n_actions: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self._buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        s  = np.stack([b[0] for b in batch])
        a  = np.array([b[1] for b in batch], dtype=np.int64)
        r  = np.array([b[2] for b in batch], dtype=np.float32)
        s2 = np.stack([b[3] for b in batch])
        d  = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self._buf)


def obs_to_vec(obv: np.ndarray) -> np.ndarray:
    # normalise continuous 0–200 px observations to [0, 1]
    return (obv.astype(np.float32) / obs_high).clip(0.0, 1.0)


def soft_update_target(tau: float = 1.0) -> None:
    with torch.no_grad():
        for tp, p in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


def train_step(discount_factor: float) -> float | None:
    if len(replay_buffer) < max(BATCH_SIZE, WARMUP_STEPS):
        return None
    s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)
    s_t  = torch.as_tensor(s,  device=device, dtype=torch.float32)
    a_t  = torch.as_tensor(a,  device=device, dtype=torch.int64)
    r_t  = torch.as_tensor(r,  device=device, dtype=torch.float32)
    s2_t = torch.as_tensor(s2, device=device, dtype=torch.float32)
    d_t  = torch.as_tensor(d,  device=device, dtype=torch.float32)

    q_sa = policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # double DQN: policy net selects action, target net evaluates
        best_a = policy_net(s2_t).argmax(1, keepdim=True)
        q_next = target_net(s2_t).gather(1, best_a).squeeze(1)
        target = r_t + discount_factor * (1.0 - d_t) * q_next

    loss = F.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return float(loss.item())


def select_action(state_vec: np.ndarray, explore_rate: float) -> int:
    if random.random() < explore_rate:
        return env.action_space.sample()
    with torch.no_grad():
        x = torch.as_tensor(state_vec, device=device, dtype=torch.float32).unsqueeze(0)
        return int(policy_net(x).argmax(dim=1).item())


def get_explore_rate(t: int) -> float:
    frac = min(1.0, t / EPSILON_DECAY_STEPS)
    return MAX_EXPLORE_RATE + frac * (MIN_EXPLORE_RATE - MAX_EXPLORE_RATE)


def save_checkpoint(episode: int) -> None:
    path = f"models_{VERSION_NAME}/checkpoint_{episode}.pt"
    torch.save({
        "episode": episode,
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_high": obs_high,
    }, path)
    print(path, "saved")


def load_checkpoint(episode: int) -> None:
    path = f"models_{VERSION_NAME}/checkpoint_{episode}.pt"
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    policy_net.load_state_dict(ckpt["policy_state_dict"])
    if "target_state_dict" in ckpt:
        target_net.load_state_dict(ckpt["target_state_dict"])
    else:
        target_net.load_state_dict(policy_net.state_dict())
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print("Loaded", path)


def simulate_parallel(n_envs: int = N_ENVS, init_env_steps: int = 0) -> None:
    envs_list = [env] + [gym.make("Pyrace-v3").unwrapped for _ in range(n_envs - 1)]
    for e in envs_list:
        e.set_view(False)

    states      = []
    ep_rewards  = [0.0] * n_envs
    prev_dists  = [0.0] * n_envs
    ep_steps    = [0]   * n_envs
    for e in envs_list:
        obv, _ = e.reset()
        states.append(obs_to_vec(obv))

    total_rewards: list[float] = []
    max_reward    = -10_000.0
    env_steps     = init_env_steps
    episodes_done = 0
    last_loss: float | None = None

    print(f"[Parallel DQN v2] {n_envs} envs  init_eps={get_explore_rate(env_steps):.3f}")

    while episodes_done < NUM_EPISODES:
        explore_rate = get_explore_rate(env_steps)

        for i, e in enumerate(envs_list):
            action = select_action(states[i], explore_rate)
            obv, reward, done, _, info = e.step(action)
            next_state = obs_to_vec(obv)

            # reward already includes speed + checkpoint bonuses from the env;
            # add only a small progress delta on top for extra density
            dist_delta = info["dist"] - prev_dists[i]
            r_train    = (float(reward) + PROGRESS_REWARD_SCALE * dist_delta) / REWARD_SCALE
            replay_buffer.push(states[i], action, r_train, next_state, bool(done))

            ep_rewards[i] += reward
            prev_dists[i]  = info["dist"]
            ep_steps[i]   += 1
            env_steps      += 1
            states[i]       = next_state

            if done or ep_steps[i] >= MAX_T:
                total_rewards.append(ep_rewards[i])
                if ep_rewards[i] > max_reward:
                    max_reward = ep_rewards[i]
                episodes_done += 1

                if episodes_done % LOG_EVERY_EPISODES == 0:
                    tail   = total_rewards[-min(100, len(total_rewards)):]
                    mean_r = sum(tail) / len(tail)
                    ls     = f"{last_loss:.4f}" if last_loss is not None else "n/a"
                    print(
                        f"[ParDQN-v2] ep {episodes_done}  mean_return(last {len(tail)})={mean_r:.1f}  "
                        f"eps={explore_rate:.3f}  steps={env_steps}  buf={len(replay_buffer)}  loss={ls}"
                    )

                if episodes_done % REPORT_EPISODES == 0:
                    plt.plot(total_rewards)
                    plt.ylabel("rewards")
                    curve_path = f"models_{VERSION_NAME}/rewards_curve_{episodes_done}.png"
                    plt.savefig(curve_path)
                    print(f"Saved learning curve: {curve_path}")
                    save_checkpoint(episodes_done)
                    plt.close()

                ep_rewards[i] = 0.0
                prev_dists[i] = 0.0
                ep_steps[i]   = 0
                obv2, _       = e.reset()
                states[i]     = obs_to_vec(obv2)

        for _ in range(TRAIN_UPDATES_PER_STEP):
            loss = train_step(DISCOUNT_FACTOR)
            if loss is not None:
                last_loss = loss
        if env_steps % TARGET_UPDATE_EVERY == 0:
            soft_update_target(tau=1.0)

    for e in envs_list[1:]:
        e.close()


def simulate(learning: bool = True, episode_start: int = 0) -> None:
    total_reward = 0.0
    total_rewards: list[float] = []
    max_reward = -10_000.0
    if not learning:
        import pygame
        env.pyrace.enable_display()
        env.pyrace.map = env.pyrace.map.convert()
        env.pyrace._car_surface = env.pyrace._car_surface.convert_alpha()
        env.pyrace.game_speed = 30
        env.set_view(True)

    for episode in range(episode_start, NUM_EPISODES + episode_start):
        if episode > episode_start:
            total_rewards.append(total_reward)

        env.memory = []
        obv, _ = env.reset()
        if not learning:
            env.pyrace.car.surface = env.pyrace._car_surface
            env.pyrace.mode = 2
        s = obs_to_vec(obv)
        total_reward = 0.0

        for t in range(MAX_T):
            a = select_action(s, 0.0 if not learning else get_explore_rate(t))
            obv, reward, done, _, info = env.step(a)
            s = obs_to_vec(obv)
            total_reward += reward

            if not learning:
                env.set_msgs([
                    "SIMULATE",
                    f"Episode: {episode}",
                    f"Time steps: {t}",
                    f'check: {info["check"]}',
                    f'dist: {info["dist"]}',
                    f'crash: {info["crash"]}',
                    f"Reward: {total_reward:.0f}",
                    f"Max Reward: {max_reward:.0f}",
                ])
                env.render()
                import pygame; pygame.event.pump()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break


def load_and_play(episode: int, learning: bool = False) -> None:
    print("Start loading checkpoint")
    load_checkpoint(episode)
    simulate(learning=learning, episode_start=episode)


if __name__ == "__main__":
    env = gym.make("Pyrace-v3").unwrapped
    print("env", type(env))
    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    NUM_ACTIONS = env.action_space.n                             # 4
    n_obs       = int(np.prod(env.observation_space.shape))     # 5
    obs_high    = env.observation_space.high.astype(np.float32) # [200, 200, 200, 200, 200]

    print(f"n_obs={n_obs}  n_actions={NUM_ACTIONS}  obs_high={obs_high}")

    MIN_EXPLORE_RATE = 0.001
    DISCOUNT_FACTOR  = 0.99
    NUM_EPISODES     = 65_000
    MAX_T            = 2_000

    policy_net    = QNetwork(n_obs, NUM_ACTIONS).to(device)
    target_net    = QNetwork(n_obs, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer     = optim.Adam(policy_net.parameters(), lr=ADAM_LR)
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

    # -------------
    WATCH = True   # True = load checkpoint and watch; False = train

    if WATCH:
        load_and_play(2000, learning=False)
    else:
        simulate_parallel()                                               # train from scratch
        # load_checkpoint(1500); simulate_parallel(init_env_steps=27_000)  # resume training
    # -------------
