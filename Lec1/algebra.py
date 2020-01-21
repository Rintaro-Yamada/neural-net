#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1回演習問題
"""
import numpy as np

########## 課題1(a)
A = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]],dtype=float)
b = np.array([1,0,1,0,1],dtype=float)

########## 課題1(b)
print(np.dot(A,b))
########## 課題1(c)
print(np.sum(A,axis =0))
print(np.sum(A,axis=1))

########## 課題1(d)-i.
a=0
for x in range(0,10):
    if x==0:
        print(a)
    else:
        a=2*a+1
        print(a)

########## 課題1(d)-ii.
a=6        
for y in range(0,10):
    if y==0:
        print(a)
    else:
        if (a%2)==0:
            a=a/2
            print(a)
        else:
            a=3*a+1
            print(a)
        