"""
DQN for Pyrace-v1 (Part 1 — replaces tabular Q in Pyrace_RL_QTable.py).

Uses experience replay, a target network (periodic hard sync), reward scaling for
stability (env rewards are 0 most steps then ±~10k at terminal events), and
gradient clipping. Episode return printed to stdout is still the raw sum of env rewards.

Run (from this directory so `import gym_race` works):
  pip install -r requirements.txt
  python Pyrace_RL_DQN.py

Baseline Q-table (for comparison):
  python Pyrace_RL_QTable.py
  - Train: in Pyrace_RL_QTable.py uncomment simulate() and comment load_and_play(...)
  - Evaluate only: use load_and_play(<episode>, learning=False) with matching checkpoints in models_QT_v02/

This DQN script:
  - Train: keep simulate() in __main__ (default); checkpoints: models_DQN_v01/checkpoint_<episode>.pt
  - Evaluate: comment simulate(), call load_and_play(<episode>, learning=False) after training

Headless servers: export SDL_VIDEODRIVER=dummy
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

# gym_race registers Pyrace-v1 in gymnasium
VERSION_NAME = "DQN_v01"

REPORT_EPISODES = 500
DISPLAY_EPISODES = 100

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 256
# Most env steps have reward 0; only crash/goal are huge — start training once buffer has one batch.
WARMUP_STEPS = 512
TRAIN_UPDATES_PER_STEP = 1  # 1 gradient update per env step (standard ratio; was 4)
HIDDEN_DIM = 256
ADAM_LR = 1e-3
# evaluate() uses ~-10k..+10k; scaling stabilizes DQN targets (logged episode return stays raw).
REWARD_SCALE = 10_000.0
GRAD_CLIP_NORM = 10.0
# Target net reduces bootstrap noise; update every N env steps (standard DQN fix).
TARGET_UPDATE_EVERY = 500
LOG_EVERY_EPISODES = 50
# Epsilon linear decay: explore fully for first EPSILON_DECAY_STEPS env steps, then anneal.
MAX_EXPLORE_RATE = 1.0
EPSILON_DECAY_STEPS = 30_000
# Small per-step reward for forward progress along track (makes signal denser).
PROGRESS_REWARD_SCALE = 0.01
# Bonus per checkpoint passed (7 checkpoints × 1500 = 10500 total shaped signal for a full lap).
CHECKPOINT_REWARD = 1500.0
# Tiny bonus per step alive — nudges agent to survive longer.
SURVIVAL_REWARD = 1.0
# Number of independent envs to run in parallel (same process, shared replay buffer).
N_ENVS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = None
policy_net = None
target_net = None
optimizer = None
replay_buffer = None
obs_high: np.ndarray | None = None


class QNetwork(nn.Module):
    """MLP: state dim -> Q(s, a) for each discrete action."""

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

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        s = np.stack([b[0] for b in batch])
        a = np.array([b[1] for b in batch], dtype=np.int64)
        r = np.array([b[2] for b in batch], dtype=np.float32)
        s2 = np.stack([b[3] for b in batch])
        d = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self._buf)


def obs_to_vec(obv: np.ndarray) -> np.ndarray:
    h = obs_high
    return (obv.astype(np.float32) / h).clip(0.0, 1.0)


def soft_update_target(tau: float = 1.0) -> None:
    """tau=1.0 => hard copy (used for periodic sync)."""
    with torch.no_grad():
        for tp, p in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


def train_step(discount_factor: float) -> float | None:
    if len(replay_buffer) < max(BATCH_SIZE, WARMUP_STEPS):
        return None
    s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)
    s_t = torch.as_tensor(s, device=device, dtype=torch.float32)
    a_t = torch.as_tensor(a, device=device, dtype=torch.int64)
    r_t = torch.as_tensor(r, device=device, dtype=torch.float32)
    s2_t = torch.as_tensor(s2, device=device, dtype=torch.float32)
    d_t = torch.as_tensor(d, device=device, dtype=torch.float32)

    q_all = policy_net(s_t)
    q_sa = q_all.gather(1, a_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: policy_net selects action, target_net evaluates it (reduces overestimation)
        best_actions = policy_net(s2_t).argmax(1, keepdim=True)
        q_next = target_net(s2_t).gather(1, best_actions).squeeze(1)
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
        q = policy_net(x)
        return int(q.argmax(dim=1).item())


def get_explore_rate(t: int) -> float:
    frac = min(1.0, t / EPSILON_DECAY_STEPS)
    return MAX_EXPLORE_RATE + frac * (MIN_EXPLORE_RATE - MAX_EXPLORE_RATE)


def get_learning_rate_tabular_schedule(t: int) -> float:
    """Kept for parity with Q-table script (not used for Adam)."""
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


def save_checkpoint(episode: int) -> None:
    path = f"models_{VERSION_NAME}/checkpoint_{episode}.pt"
    torch.save(
        {
            "episode": episode,
            "policy_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_high": obs_high,
        },
        path,
    )
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


def simulate(learning: bool = True, episode_start: int = 0) -> None:
    discount_factor = DISCOUNT_FACTOR
    total_reward = 0.0
    total_rewards: list[float] = []
    max_reward = -10_000.0
    if not learning:
        env.pyrace.enable_display()
        env.set_view(True)
    env_steps = 0
    last_loss: float | None = None

    for episode in range(episode_start, NUM_EPISODES + episode_start):
        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel("rewards")
                curve_path = f"models_{VERSION_NAME}/rewards_curve_{episode}.png"
                plt.savefig(curve_path)
                print(f"Saved learning curve: {curve_path}")
                mem_file = f"models_{VERSION_NAME}/memory_{episode}"
                env.save_memory(mem_file)
                save_checkpoint(episode)
                plt.close()

        show_this_episode = not learning  # training always headless; display only during evaluation

        env.memory = []
        obv, _ = env.reset()
        state_0 = obs_to_vec(obv)
        total_reward = 0.0
        prev_dist = 0.0
        if not learning:
            env.pyrace.mode = 2

        explore_rate = get_explore_rate(env_steps)
        eps_used_this_episode = explore_rate

        for t in range(MAX_T):
            action = select_action(state_0, explore_rate if learning else 0.0)
            obv, reward, done, _, info = env.step(action)
            state = obs_to_vec(obv)
            env.remember(state_0, action, reward, state, done)
            total_reward += reward

            if learning:
                # Reward shaping: small bonus for forward progress (denser signal).
                dist_delta = info["dist"] - prev_dist
                r_train = (float(reward) + PROGRESS_REWARD_SCALE * dist_delta) / REWARD_SCALE
                replay_buffer.push(state_0, action, r_train, state, bool(done))
                env_steps += 1
                explore_rate = get_explore_rate(env_steps)
                for _ in range(TRAIN_UPDATES_PER_STEP):
                    loss = train_step(discount_factor)
                    if loss is not None:
                        last_loss = loss
                if env_steps % TARGET_UPDATE_EVERY == 0:
                    soft_update_target(tau=1.0)

            prev_dist = info["dist"]
            state_0 = state

            if show_this_episode:
                env.set_msgs(
                    [
                        "SIMULATE",
                        f"Episode: {episode}",
                        f"Time steps: {t}",
                        f'check: {info["check"]}',
                        f'dist: {info["dist"]}',
                        f'crash: {info["crash"]}',
                        f"Reward: {total_reward:.0f}",
                        f"Max Reward: {max_reward:.0f}",
                    ]
                )
                env.render()
            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break

        _ = get_learning_rate_tabular_schedule(episode)
        if learning and (episode + 1) % LOG_EVERY_EPISODES == 0 and total_rewards:
            tail = total_rewards[-min(100, len(total_rewards)) :]
            mean_r = sum(tail) / len(tail)
            ls = f"{last_loss:.4f}" if last_loss is not None else "n/a"
            print(
                f"[DQN] episode {episode + 1}  mean_return(last {len(tail)})={mean_r:.1f}  "
                f"eps_used={eps_used_this_episode:.3f}  buf={len(replay_buffer)}  loss={ls}"
            )


def simulate_parallel(n_envs: int = N_ENVS, init_env_steps: int = 0) -> None:
    """Train with n_envs independent environments sharing one replay buffer.

    All envs run in the same process (no multiprocessing / GIL issues with pygame).
    Each env steps sequentially per iteration, but we collect n_envs experiences
    before each gradient update — effectively n_envs times more data per wall-clock step.

    init_env_steps: set >0 when resuming so epsilon starts at the right value.
    """
    envs_list = [env] + [gym.make("Pyrace-v1").unwrapped for _ in range(n_envs - 1)]
    for e in envs_list:
        e.set_view(False)

    # Per-env episode state
    states       = []
    ep_rewards   = [0.0] * n_envs
    prev_dists   = [0.0] * n_envs
    prev_checks  = [0]   * n_envs  # last checkpoint index seen per env
    ep_steps     = [0]   * n_envs
    for e in envs_list:
        obv, _ = e.reset()
        states.append(obs_to_vec(obv))

    total_rewards: list[float] = []
    max_reward   = -10_000.0
    env_steps    = init_env_steps
    episodes_done = 0
    last_loss: float | None = None

    print(f"[Parallel DQN] Starting with {n_envs} envs  init_eps={get_explore_rate(env_steps):.3f}")

    while episodes_done < NUM_EPISODES:
        explore_rate = get_explore_rate(env_steps)

        for i, e in enumerate(envs_list):
            action = select_action(states[i], explore_rate)
            obv, reward, done, _, info = e.step(action)
            next_state = obs_to_vec(obv)

            dist_delta   = info["dist"] - prev_dists[i]
            # Checkpoint bonus: reward agent for each new checkpoint reached this episode.
            # max(0,...) guards against the counter resetting to 0 on episode end.
            check_bonus  = max(0, info["check"] - prev_checks[i]) * CHECKPOINT_REWARD
            r_train      = (float(reward) + check_bonus + SURVIVAL_REWARD
                            + PROGRESS_REWARD_SCALE * dist_delta) / REWARD_SCALE
            replay_buffer.push(states[i], action, r_train, next_state, bool(done))

            ep_rewards[i]  += reward
            prev_dists[i]   = info["dist"]
            prev_checks[i]  = max(prev_checks[i], info["check"])
            ep_steps[i]    += 1
            env_steps       += 1
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
                        f"[ParDQN] ep {episodes_done}  mean_return(last {len(tail)})={mean_r:.1f}  "
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

                ep_rewards[i]  = 0.0
                prev_dists[i]  = 0.0
                prev_checks[i] = 0
                ep_steps[i]    = 0
                obv2, _        = e.reset()
                states[i]      = obs_to_vec(obv2)

        for _ in range(TRAIN_UPDATES_PER_STEP):
            loss = train_step(DISCOUNT_FACTOR)
            if loss is not None:
                last_loss = loss
        if env_steps % TARGET_UPDATE_EVERY == 0:
            soft_update_target(tau=1.0)

    # Clean up extra envs
    for e in envs_list[1:]:
        e.close()


def load_and_play(episode: int, learning: bool = False) -> None:
    print("Start loading checkpoint")
    load_checkpoint(episode)
    simulate(learning=learning, episode_start=episode)


if __name__ == "__main__":
    env = gym.make("Pyrace-v1").unwrapped
    print("env", type(env))
    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    NUM_BUCKETS = tuple(
        (env.observation_space.high + np.ones(env.observation_space.shape)).astype(int)
    )
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    n_obs = int(np.prod(env.observation_space.shape))
    obs_high = env.observation_space.high.astype(np.float32)

    print(NUM_BUCKETS, NUM_ACTIONS, STATE_BOUNDS, "n_obs", n_obs)

    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DISCOUNT_FACTOR = 0.99
    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0
    print("DECAY_FACTOR", DECAY_FACTOR)

    NUM_EPISODES = 65_000
    MAX_T = 2000

    policy_net = QNetwork(n_obs, NUM_ACTIONS).to(device)
    target_net = QNetwork(n_obs, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=ADAM_LR)
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

    # -------------
    load_and_play(3500, learning=False)
    # load_checkpoint(1500); simulate_parallel(init_env_steps=27_000)  # resume training
    # simulate_parallel()          # train from scratch
    # -------------
