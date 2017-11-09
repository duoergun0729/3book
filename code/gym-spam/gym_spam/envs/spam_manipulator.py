#-*- coding:utf-8 –*-


import numpy as np
import re
import random



class Spam_Manipulator(object):
    def __init__(self):
        self.dim = 0
        self.name=""

#常见免杀动作：

    # 大小写混淆
    # 增加TAB
    # 增加回车
    # 增加换行符
    ACTION_TABLE = {
    'addTab': 'addTab',
    'addEnter': 'addEnter',
    'confusionCase':'confusionCase',
    'lineBreak': 'lineBreak',
    'addHyphen': 'addHyphen',
    'doubleChar': 'doubleChar',
    }

    def confusionCase(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'\b\w+\b', str, re.M | re.I)
        if matchObjs:
            #选择替换的单词
            modify_word=random.choice(matchObjs)
            #生成替换的内容
            modify_word_swapcase=modify_word.swapcase()

            #替换
            str=re.sub(modify_word, modify_word_swapcase, str,count=random.randrange(1,3))

        return str

    def lineBreak(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            # 选择替换的字符
            modify_char = random.choice(matchObjs)
            # 生成替换的内容
            modify_char_lb = "{}/".format(modify_char)
            #print modify_char_lb

            # 替换
            str = re.sub(modify_char, modify_char_lb, str, count=random.randrange(1, 3))

        return str

    def addTab(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            #选择替换的字符
            modify_char=random.choice(matchObjs)
            #生成替换的内容
            modify_char_tab="   {}".format(modify_char)

            #替换
            str=re.sub(modify_char, modify_char_tab, str,count=random.randrange(1,3))

        return str

    def addHyphen(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            # 选择替换的字符
            modify_char = random.choice(matchObjs)
            # 生成替换的内容
            modify_char_lb = "{}-".format(modify_char)
            #print modify_char_lb

            # 替换
            str = re.sub(modify_char, modify_char_lb, str, count=random.randrange(1, 3))

        return str

    def addEnter(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            #选择替换的字符
            modify_char=random.choice(matchObjs)
            #生成替换的内容
            modify_char_enter="\\r\\n{}".format(modify_char)

            #替换
            str=re.sub(modify_char, modify_char_enter, str,count=1)

        return str

    def doubleChar(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            #选择替换的字符
            modify_char=random.choice(matchObjs)
            #生成替换的内容
            modify_char_enter="{}{}".format(modify_char,modify_char)

            #替换
            str=re.sub(modify_char, modify_char_enter, str,count=random.randrange(1,3))

        return str

    def modify(self,str, _action, seed=6):

        print "Do action :%s" % _action
        action_func=Spam_Manipulator().__getattribute__(_action)

        return action_func(str,seed)


if __name__ == '__main__':
    f=Spam_Manipulator()
    raw="thank you ,your email address was obtained from a purchased list ," \
        "reference # 2020 mid = 3300 . if you wish to unsubscribe"
    a=f.modify(raw,"addEnter")
    print "addEnter:"
    print a

    a=f.modify(raw,"addTab")
    print "addTab:"
    print a

    a=f.modify(raw,"lineBreak")
    print "lineBreak:"
    print a

    a=f.modify(raw,"confusionCase")
    print "confusionCase:"
    print a

    a=f.modify(raw,"addHyphen")
    print "addHyphen:"
    print a

    a=f.modify(raw,"doubleChar")
    print "doubleChar:"
    print a
