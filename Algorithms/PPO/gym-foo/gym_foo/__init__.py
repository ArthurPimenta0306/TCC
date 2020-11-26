from gym.envs.registration import register

register(
    id='Pendulum-tucho-v0',
    entry_point='gym_foo.envs:PendulumEnv',
    max_episode_steps=200,
)
register(
    id='DroneRoll-v0',
    entry_point='gym_foo.envs:EnvRoll',
    max_episode_steps=200,
)
register(
    id='DronePitch-v0',
    entry_point='gym_foo.envs:EnvPitch',
    max_episode_steps=200,
)
register(
    id='DroneYaw-v0',
    entry_point='gym_foo.envs:EnvYaw',
    max_episode_steps=200,
)
register(
    id='foo-extrahard-v0',
    entry_point='gym_foo.envs:FooExtraHardEnv',
)