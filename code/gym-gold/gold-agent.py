#-*- coding:utf-8 –*-
import gym
import time
import random
import gym_gold.envs.goldEnv

states = [1, 2, 3, 4, 5, 6, 7, 8]
actions = ['n', 'e', 's', 'w']

def greedy(Q,state):
    amax = 0
    key = "%d_%s" % (state, actions[0])

    qmax = Q[key]
    for i in range(len(actions)):
        key = "%d_%s" % (state, actions[i])
        q = Q[key]
        if qmax < q:
            qmax = q
            amax = i
    return actions[amax]

def epsilon_greedy(Q, state, epsilon):
    amax = 0
    key = "%d_%s"%(state, actions[0])
    qmax = Q[key]
    for i in range(len(actions)):
        key = "%d_%s"%(state, actions[i])
        q = Q[key]
        if qmax < q:
            qmax = q
            amax = i

    pro = [0.0 for i in range(len(actions))]
    pro[amax] += 1-epsilon
    for i in range(len(actions)):
        pro[i] += epsilon/len(actions)


    r = random.random()
    s = 0.0
    for i in range(len(actions)):
        s += pro[i]
        if s>= r: return actions[i]
    return actions[len(actions)-1]


if __name__ == '__main__':
    env = gym.make('Gold-v1')
    alpha=0.1
    gamma=0.5
    epsilon=0.1
    random.seed(0)

    Q = dict()

    print int(4.1)
    print int(3.9)


    for s in states:
        for a in actions:
            key = "%d_%s"%(s,a)
            Q[key]=0

    gold=0

    for episode in range(10):
        s0 = env.reset()
        a0 = greedy(Q,s0)

        #狗屎运 初始化就拿到金币了
        if s0 == 7 :
            continue

        #print("Episode start at state:{}".format(s0))
        for t in range(50):
            observation, reward, done, info = env.step(a0)
            s1=observation
            #贪婪算法
            #a1 = greedy(Q, s1)
            #epsilon贪婪算法
            a1 = epsilon_greedy(Q,s1,epsilon)

            key0=   "%d_%s" % (s0, a0)
            key1 = "%d_%s" % (s1, a1)
            #更新Q函数
            Q[key0] = Q[key0] + alpha * (reward + gamma * Q[key1] - Q[key0])
            a0=a1
            s0=s1
            if done and s1==7 :
                print("Get Gold {}th Episode finished after {} timesteps ".format(episode,t+1))
                gold+=1
                break
            if done :
                #print("Episode finished after {} timesteps ".format(t + 1))
                break


    print "episode:{} get gold:{}".format(episode,gold)