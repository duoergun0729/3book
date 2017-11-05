#-*- coding:utf-8 –*-

import numpy as np




class Features(object):
    def __init__(self):
        self.dim = 0
        self.name=""

    def str_len(self,str):
        return len(str)

    def extract(self,str):

        featurevectors = [
            [self.str_len(str)],
            [self.str_len(str)]
        ]
        return np.concatenate(featurevectors)


if __name__ == '__main__':
    f=Features()
    a=f.extract("alert()")
    print a.shape
    print a.shape[0]