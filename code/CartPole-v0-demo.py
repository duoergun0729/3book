# -*- coding: utf-8 -*-
import gym
import time
env = gym.make('CartPole-v0')
observation = env.reset()
print observation

print "env actionspace："
print(env.action_space)

print "env observationspace："
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

count = 0
for t in range(100):
    #随机选择一个动作
    action = env.action_space.sample()
    #执行动作 获取环境反馈
    observation, reward, done, info = env.step(action)
    #如果玩死了就退出
    if done:
        break
    env.render()
    count+=1
    time.sleep(0.2)
print(count)
