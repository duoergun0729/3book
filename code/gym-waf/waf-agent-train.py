#-*- coding:utf-8 –*-
import gym
import time
import random
import gym_waf.envs.wafEnv
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, ELU, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop


from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SarsaAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'Waf-v0'

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

    memory = SequentialMemory(limit=32, ignore_episode_boundaries=False, window_length=window_length)

    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=16,
                     enable_double_dqn=True, enable_dueling_network=True, dueling_type='avg',
                     target_model_update=1e-2, policy=policy, batch_size=16)

    agent.compile(RMSprop(lr=1e-3), metrics=['mae'])

    # play the game. learn something!
    agent.fit(env, nb_steps=rounds, visualize=False, verbose=2)

    print "#################Start Test%################"

    agent.test(env, nb_episodes=100)

    return agent, model


if __name__ == '__main__':
    agent1, model1= train_dqn_model([5, 2], rounds=1000)
    model1.save('waf-v0.h5', overwrite=True)




