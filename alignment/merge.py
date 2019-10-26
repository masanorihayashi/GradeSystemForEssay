#encoding:utf-8

import sys
argvs = sys.argv

f_ori = open(argvs[1], "r")
f_cor = open(argvs[2], "r")

for ori_line, cor_line in zip(f_ori.readlines(), f_cor.readlines()):
    ori = ori_line.rstrip()
    cor = cor_line.rstrip()
    if ori == "":
        ori == "."
    if cor == "":
        cor == "."
    print ori
    print cor
