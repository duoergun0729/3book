#-*- coding:utf-8 –*-

import numpy as np
import re

#<embed src="data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==">
#a="get";b="URL(ja\"";c="vascr";d="ipt:ale";e="rt('XSS');\")";eval(a+b+c+d+e);
#"><script>alert(String.fromCharCode(66, 108, 65, 99, 75, 73, 99, 101))</script>
#<input onblur=write(XSS) autofocus><input autofocus>
#<math><a xlink:href="//jsfiddle.net/t846h/">click
#<h1><font color=blue>hellox worldss</h1>
#LOL<style>*{/*all*/color/*all*/:/*all*/red/*all*/;/[0]*IE,Safari*[0]/color:green;color:bl/*IE*/ue;}</style>


class Waf_Check(object):
    def __init__(self):
        self.name="Waf_Check"
        self.regXSS=r'(prompt|alert|confirm|expression])' \
                    r'|(javascript|script|eval)' \
                    r'|(onload|onerror|onfocus|onclick|ontoggle|onmousemove|ondrag)' \
                    r'|(String.fromCharCode)' \
                    r'|(;base64,)' \
                    r'|(onblur=write)' \
                    r'|(xlink:href)' \
                    r'|(color=)'
        #self.regXSS = r'javascript'



    def check_xss(self,str):
        isxss=False

        #忽略大小写
        if re.search(self.regXSS,str,re.IGNORECASE):
            isxss=True

        return isxss


if __name__ == '__main__':
    waf=Waf_Check()
    #checklistfile="../../xss-samples.txt"
    checklistfile = "../../xss-samples-all.txt"

    with open(checklistfile) as f:
        for line in f:
            line=line.strip('\n')
            #print line
            if waf.check_xss(line):
                print "Match waf rule :"
                print line
