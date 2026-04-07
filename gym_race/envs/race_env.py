import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D, PyRace2DV3

class RaceEnv(gym.Env):
    metadata = {'render_modes' : ['human'], 'render_fps' : 30}
    def __init__(self, render_mode="human", ):
        print("init")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=int)
        self.is_view = False  # headless by default; call set_view(True) + pyrace.enable_display() for rendering
        self.pyrace = PyRace2D(False)
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # PyRace2D.mode must be int 0/1/2 (view). Do not pass render_mode ("human") — that breaks m/p keys.
        prev_mode = self.pyrace.mode
        if not isinstance(prev_mode, int):
            prev_mode = 0
        # Mode 1 = full black debug view; keep it from sticking across episodes during training.
        if prev_mode == 1:
            prev_mode = 0
        self.msgs = []
        # Reset car in-place — avoids pygame.init() + image I/O on every episode.
        self.pyrace.reset_state(mode=prev_mode)
        obs = self.pyrace.observe()
        return np.array(obs), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return np.array(obs), reward, done, False, {'dist':self.pyrace.car.distance, 'check':self.pyrace.car.current_check, 'crash': not self.pyrace.car.is_alive}

    # def render(self, close=False , msgs=[], **kwargs): # gymnasium.render() does not accept other keyword arguments
    def render(self): # gymnasium.render() does not accept other keyword arguments
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


class RaceEnvV3(gym.Env):
    """Gym wrapper for PyRace2DV3 (Pyrace-v3).

    Differences vs RaceEnv (Pyrace-v1):
      - action_space: Discrete(4) — adds BRAKE
      - observation_space: Box(0, 200, float32) — continuous radar distances in pixels
      - reward: dense per-step signal from PyRace2DV3.evaluate()
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode="human"):
        print("init")
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.full(5, 200.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.is_view = False
        self.pyrace = PyRace2DV3(False)
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        prev_mode = self.pyrace.mode
        if not isinstance(prev_mode, int):
            prev_mode = 0
        if prev_mode == 1:
            prev_mode = 0
        self.msgs = []
        self.pyrace.reset_state(mode=prev_mode)
        obs = self.pyrace.observe()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return (
            np.array(obs, dtype=np.float32),
            reward,
            done,
            False,
            {'dist': self.pyrace.car.distance, 'check': self.pyrace.car.current_check, 'crash': not self.pyrace.car.is_alive},
        )

    def render(self):
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
