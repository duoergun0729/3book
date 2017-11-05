#-*- coding:utf-8 –*-

import numpy as np
import re



class Waf_Check(object):
    def __init__(self):
        self.name="Waf_Check"
        self.regXSS=r'(prompt|alert|confirm])' \
                    r'|(javascript|script)' \
                    r'|(onload|onerror|onfocus|onclick|ontoggle|onmousemove|ondrag)'
        #self.regXSS = r'javascript'



    def check_xss(self,str):
        isxss=False

        #忽略大小写
        if re.search(self.regXSS,str,re.IGNORECASE):
            isxss=True

        return isxss


if __name__ == '__main__':
    waf=Waf_Check()
    checklistfile="../../xss-samples.txt"

    with open(checklistfile) as f:
        for line in f:
            line=line.strip('\n')
            #print line
            if waf.check_xss(line):
                print "Match waf rule :"
                print line