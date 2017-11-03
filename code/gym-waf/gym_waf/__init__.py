from gym.envs.registration import register


register(
    id='Waf-v0',
    entry_point='gym_waf.envs.wafEnv:WafEnv_v0',
)



