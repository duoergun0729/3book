#-*- coding:utf-8 –*-
import numpy
import random
from gym import spaces
import gym
from features import Features
from waf import Waf_Check
from xss_manipulator import Xss_Manipulator
#新版接口
from sklearn.model_selection import train_test_split

#samples_file="xss-samples.txt"
samples_file="xss-samples-all.txt"
samples=[]
with open(samples_file) as f:
    for line in f:
        line = line.strip('\n')
        print "Add xss sample:" + line
        samples.append(line)

# 划分训练和测试集合
samples_train, samples_test = train_test_split(samples, test_size=0.4)


ACTION_LOOKUP = {i: act for i, act in enumerate(Xss_Manipulator.ACTION_TABLE.keys())}


class WafEnv_v0(gym.Env):
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
        self.features_extra=Features()
        self.waf_checker=Waf_Check()
        #根据动作修改当前样本免杀
        self.xss_manipulatorer= Xss_Manipulator()

        self._reset()



    def _step(self, action):

        r=0
        is_gameover=False
        #print "current sample:%s" % self.current_sample

        _action=ACTION_LOOKUP[action]
        #print "action is %s" % _action

        self.current_sample=self.xss_manipulatorer.modify(self.current_sample,_action)
        #print "change current sample to %s" % self.current_sample

        if not self.waf_checker.check_xss(self.current_sample):
            #给奖励
            r=10
            is_gameover=True
            print "Good!!!!!!!avoid waf:%s" % self.current_sample

        self.observation_space=self.features_extra.extract(self.current_sample)

        return self.observation_space, r,is_gameover,{}


    def _reset(self):
        self.current_sample=random.choice(samples_train)
        print "reset current_sample=" + self.current_sample

        self.observation_space=self.features_extra.extract(self.current_sample)
        return self.observation_space


    def render(self, mode='human', close=False):
        return