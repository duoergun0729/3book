#-*- coding:utf-8 –*-

import numpy as np




class Features(object):
    def __init__(self):
        self.dim = 0
        self.name=""
        self.dtype=np.float32

    def byte_histogram(self,str):
        #bytes=np.array(list(str))
        bytes=[ord(ch) for ch in list(str)]
        #print bytes

        h = np.bincount(bytes, minlength=256)
        return np.concatenate([
            [h.sum()],  # total size of the byte stream
            h.astype(self.dtype).flatten() / h.sum(),  # normalized the histogram
        ])

    def extract(self,str):

        featurevectors = [
            [self.byte_histogram(str)]
        ]
        return np.concatenate(featurevectors)


if __name__ == '__main__':
    f=Features()
    a=f.extract("alert()")
    print a
    print a.shape
    #print a.shape[0]