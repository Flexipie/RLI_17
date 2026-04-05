from gymnasium.envs.registration import register

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)
