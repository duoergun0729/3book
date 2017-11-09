#-*- coding:utf-8 –*-
import numpy
import random
from gym import spaces
import gym
from features import Features
from spam import Spam_Check
from spam_manipulator import Spam_Manipulator
#新版接口
from sklearn.model_selection import train_test_split

from spam import load_all_spam,mode_file,vocabulary_file


# 划分训练和测试集合
samples_train, samples_test=load_all_spam()



ACTION_LOOKUP = {i: act for i, act in enumerate(Spam_Manipulator.ACTION_TABLE.keys())}


class SpamEnv_v0(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))

        #xss样本特征集合
        #self.samples=[]
        #当前处理的样本
        self.current_sample=""
        #self.current_state=0
        self.features_extra=Features(vocabulary_file)
        self.spam_checker=Spam_Check()
        #根据动作修改当前样本免杀
        self.spam_manipulatorer= Spam_Manipulator()

        self._reset()



    def _step(self, action):

        r=0
        is_gameover=False
        #print "current sample:%s" % self.current_sample

        _action=ACTION_LOOKUP[action]
        #print "action is %s" % _action

        self.current_sample=self.spam_manipulatorer.modify(self.current_sample,_action)
        #print "change current sample to %s" % self.current_sample
        self.observation_space = self.features_extra.extract(self.current_sample)

        if self.spam_checker.check_spam(self.observation_space) < 1.0:
            #给奖励
            r=10
            is_gameover=True
            print "Good!!!!!!!avoid spam detect:%s" % self.current_sample


        return self.observation_space, r,is_gameover,{}


    def _reset(self):
        self.current_sample=random.choice(samples_train)
        #print "reset current_sample=" + self.current_sample

        self.observation_space=self.features_extra.extract(self.current_sample)
        return self.observation_space


    def render(self, mode='human', close=False):
        return