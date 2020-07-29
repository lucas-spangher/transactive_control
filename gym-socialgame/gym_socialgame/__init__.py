from gym.envs.registration import register

register(
    id='socialgame-v0',
    entry_point='gym_socialgame.envs:SocialGameEnv',
)