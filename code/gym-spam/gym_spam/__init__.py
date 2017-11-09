from gym.envs.registration import register


register(
    id='Spam-v0',
    entry_point='gym_spam.envs.spamEnv:SpamEnv_v0',
)



