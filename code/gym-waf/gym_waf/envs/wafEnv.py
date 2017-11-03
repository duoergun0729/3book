#-*- coding:utf-8 –*-
import numpy
import random
from gym import spaces
import gym


class WafEnv_v0(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):

        self.states = [1,2,3,4,5,6,7,8]

        self.terminate_states = dict()
        self.terminate_states[6] = 1
        self.terminate_states[7] = 1
        self.terminate_states[8] = 1

        self.actions = ['n','e','s','w']

        self.rewards = dict();
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        self.t = dict();
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 8
        self.t['5_w'] = 4

        self.gamma = 0.8
        self.viewer = None
        self.state = None

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions
    def getTerminate_states(self):
        return self.terminate_states
    def setAction(self,s):
        self.state=s

    def _step(self, action):

        state = self.state
        #判断是否游戏结束
        if state in self.terminate_states:
            return state, 0, True, {}
        key = "%d_%s"%(state, action)
        #查找状态迁移表
        if key in self.t:
            next_state = self.t[key]
        else:
            #查不到就状态不变
            next_state = state
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]


        return next_state, r,is_terminal,{}
    def _reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state
    def render(self, mode='human', close=False):
        return