#-*- coding:utf-8 –*-
import gym
import time
import random
import gym_waf.envs.wafEnv
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, ELU, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop


from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SarsaAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


from gym_waf.envs.wafEnv  import samples_test,samples_train
from gym_waf.envs.features import Features
from gym_waf.envs.waf import Waf_Check
from gym_waf.envs.xss_manipulator import Xss_Manipulator

from keras.callbacks import TensorBoard

ENV_NAME = 'Waf-v0'
#尝试的最大次数
nb_max_episode_steps_train=50
nb_max_episode_steps_test=3

ACTION_LOOKUP = {i: act for i, act in enumerate(Xss_Manipulator.ACTION_TABLE.keys())}

def generate_dense_model(input_shape, layers, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dropout(0.1))

    for layer in layers:
        model.add(Dense(layer))
        model.add(BatchNormalization())
        model.add(ELU(alpha=1.0))

    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    return model


def train_dqn_model(layers, rounds=10000):

    env = gym.make(ENV_NAME)
    env.seed(1)
    nb_actions = env.action_space.n
    window_length = 1

    print "nb_actions:"
    print nb_actions
    print "env.observation_space.shape:"
    print env.observation_space.shape


    model = generate_dense_model((window_length,) + env.observation_space.shape, layers, nb_actions)

    policy = EpsGreedyQPolicy()

    memory = SequentialMemory(limit=256, ignore_episode_boundaries=False, window_length=window_length)

    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=16,
                     enable_double_dqn=True, enable_dueling_network=True, dueling_type='avg',
                     target_model_update=1e-2, policy=policy, batch_size=16)

    agent.compile(RMSprop(lr=1e-3), metrics=['mae'])

    #tb_cb = TensorBoard(log_dir='/tmp/log', write_images=1, histogram_freq=1)
    #cbks = [tb_cb]
    # play the game. learn something!
    #nb_max_episode_steps 一次学习周期中最大步数
    agent.fit(env, nb_steps=rounds, nb_max_episode_steps=nb_max_episode_steps_train,visualize=False, verbose=2)

    #print "#################Start Test%################"

    #agent.test(env, nb_episodes=100)

    test_samples=samples_test

    features_extra = Features()
    waf_checker = Waf_Check()
    # 根据动作修改当前样本免杀
    xss_manipulatorer = Xss_Manipulator()

    success=0
    sum=0

    shp = (1,) + tuple(model.input_shape[1:])

    for sample in samples_test:
        #print sample
        sum+=1

        for _ in range(nb_max_episode_steps_test):

            if not waf_checker.check_xss(sample) :
                success+=1
                print sample
                break

            f = features_extra.extract(sample).reshape(shp)
            act_values = model.predict(f)
            action=np.argmax(act_values[0])
            sample=xss_manipulatorer.modify(sample,ACTION_LOOKUP[action])

    print "Sum:{} Success:{}".format(sum,success)

    return agent, model


if __name__ == '__main__':
    agent1, model1= train_dqn_model([5, 2], rounds=1000)
    model1.save('waf-v0.h5', overwrite=True)




