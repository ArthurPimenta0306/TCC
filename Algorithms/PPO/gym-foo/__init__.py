from gym.envs.registration import register

register(
    id='Pendulum-tucho-v0',
    entry_point='gym_foo.envs:PendulumEnv',
    max_episode_steps=200,
)
register(
    id='Drone-v0',
    entry_point='gym_foo.envs:Env',
    max_episode_steps=200,
)
register(
    id='foo-extrahard-v0',
    entry_point='gym_foo.envs:FooExtraHardEnv',
)