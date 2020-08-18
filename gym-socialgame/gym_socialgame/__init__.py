from gym.envs.registration import register

register(
    id='socialgame-v0',
    entry_point='gym_socialgame.envs:SocialGameEnv',
)

register(
    id='socialgame-hourly',
    entry_point='gym_socialgame.envs:SocialGameEnvHourly',
)

register(
    id='socialgame-monthly',
    entry_point='gym_socialgame.envs:SocialGameEnvMonthly',
)