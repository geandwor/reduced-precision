#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:19:24 2018

@author: uic-cs
"""

"""
Created on Tue Jun 19 14:22:57 2018

@author: uic-cs
EXPERIMENTS ON THE ERROR PROPAGATION WITH DIFFERENCE RANGE OF VALUES
with tanh activation function
tensorflow tanh doesn't support bfloat16
for datatype bfloat16, numpy.tanh is used to calculate the tanhmoid in float32, and then typecast to bfloat16
"""

import tensorflow as tf
import numpy as np
#import math

mini = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,100]
maxi = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,100,180]
leng =len(mini)
shape1 = [2,2]
shape2 = [2,2]
ite = 100

meanbf16 = []
stdbf16=[]
meanfl16=[]
stdfl16=[]
"""add code here"""
#a = tf.Variable(tf.random_uniform(shape1,minval=50,maxval = 255,dtype = tf.float32), tf.float32)
#b = tf.Variable(tf.random_uniform(shape2,minval = 100, maxval=255, dtype=tf.float32), tf.float32)
"""
tensorflow.tanhmoid doesn't support bfloat16, so I wrote a function to calculate the value in bf16
using numpy exponential
"""
def nptanh(z):
    zfloat32 = np.float32(z)
    s = np.tanh(zfloat32)
    return s


        
"""summation of the elements of the difference matrix and get the square root"""
#assume the List is either vector or two dimention matrix
def summation(ndarray):
    s = float(0)
    m,n = ndarray.shape
    for i in range(m):
        for j in range(n):
            #summation of diagnoal elements only
            if i==j:
                s +=ndarray[i][j]**2
                  
    return s

amatrix = tf.placeholder(tf.float16,shape1)
bmatrix = tf.placeholder(tf.float16,shape2)
amatrix0 = tf.placeholder(tf.bfloat16,shape1)
bmatrix0 = tf.placeholder(tf.bfloat16,shape2)
#bfloat16
c0 = tf.matmul(amatrix0,bmatrix0)
#float16
c = tf.matmul(amatrix,bmatrix)
amatrix1 = tf.placeholder(tf.float32, shape1)
bmatrix1 = tf.placeholder(tf.float32, shape2)
#float32
c1 = tf.matmul(amatrix1,bmatrix1)
#scc1 = tf.subtract(c,c1)
init = tf.global_variables_initializer() 
with tf.Session() as sess:
    sess.run(init)
    for i in range(11):
        print("iteration i{0}\n".format(i))
   
        d0 =[]
        d01=[]
        d1 =[]
        d11=[]
        d2 =[]
        d3 =[]
        d4 =[]
        d5 =[]
    
        for j in range(ite):
            a = tf.random_uniform(shape1,minval=mini[i],maxval = maxi[i],dtype = tf.float32)
            b = tf.random_uniform(shape2,minval=mini[i],maxval = maxi[i],dtype = tf.float32)
            """a1 and b1 are in float32 format"""
            a1 = sess.run(a)
            b1 = sess.run(b)
            """calculate the multiplication"""
            """1 multiplication before down cast"""
            cc = sess.run(c, feed_dict= {amatrix:a1,bmatrix:b1})
            cc0 = sess.run(c0,feed_dict={amatrix0:a1,bmatrix0:b1})
            '''calculation in tanh only diagnoal elements are considered for further calculation'''
            cc_tanh = sess.run(tf.tanh(cc))
            cc0_tanh = sess.run(tf.cast(nptanh(cc0),tf.bfloat16))
            """upcast to float32 from calcualted in 16 bit"""
            cc_tanhcast = sess.run(tf.cast(cc_tanh,tf.float32))
            cc0_tanhcast = sess.run(tf.cast(cc0_tanh,tf.float32))
            '''calculation in tanh only diagnol elements are considered for further calculation'''
            cc1 = sess.run(c1, feed_dict= {amatrix1:a1,bmatrix1:b1})
            cc1_tanh = sess.run(tf.tanh(cc1))
            #cc1_tanh_16 =sess.run(tf.cast(cc1_tanh,tf.float16))
           # cc1_tanh_bf16 = sess.run(tf.cast(cc1_tanh, tf.bfloat16))
            """difference
            "1 difference for multiplication before downcast then result upcast"""
            diffcc0cast_tanh =cc1_tanh-cc0_tanhcast
            #subtract in tensorflow in float32
            #diffcc0castf_tanh=sess.run(tf.subtract(cc1_tanh,cc0_tanhcast))
            diffcccast_tanh = cc1_tanh - cc_tanhcast
            #subtract in tensorflow in float32
            #diffcccastf_tanh=sess.run(tf.subtract(cc1_tanh,cc_tanhcast))
            """"2 difference for multiplication after downcast then result upcast"""
            """3 difference for multiplication result down cast to float16"""
            #diffcc1_16_tanh = cc1_tanh_16 - cc_tanh
            """4 difference for multilplcation result down cast to bfloat16"""
            #diffcc1_bf16_tanh = cc1_tanh_bf16 - cc0_tanh
        
            d0.append(np.sqrt(summation(diffcc0cast_tanh)))
            #square root in tensorflow in float32
            #d01.append(sess.run(tf.sqrt(summation(diffcc0cast_tanh))))
            d1.append(np.sqrt(summation(diffcccast_tanh)))
            #square root in tensorflow in float32
            #d11.append(sess.run(tf.sqrt(summation(diffcccast_tanh))))
            #d2.append(np.sqrt(summation(diffcc1_16_tanh)))
            #d3.append(np.sqrt(summation(diffcc1_bf16_tanh)))
            #square root in tensorflow in float32
            #d4.append(sess.run(tf.sqrt(summation(diffcc0castf_tanh))))
            #square root in tensorflow in float32
            #d5.append(sess.run(tf.sqrt(summation(diffcccastf_tanh))))
    print("--Iteration {0}\n".format(ite))
    print("--min {0} max {1}\n".format(mini[i],maxi[i]))
    meanbf16.append(np.mean(d0))
    stdbf16.append(np.std(d0))
    meanfl16.append(np.mean(d1))
    stdfl16.append(np.std(d1))
    print("average difference in tanh from bf16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d0),np.std(d0)))
    #the calculation results of commented out two lines are the same as the line above
    #print("average difference in tanhmoid with sqrt in tensorflow from bf16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d01),np.std(d01)))
    #print("average difference in tanhmoid with subtraction and sqrt in tensorflow from bf16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d4),np.std(d4)))
    print("average difference in tanh from fl16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d1),np.std(d1)))
    #the calculation results of commented out two lines are the same as the line above
    #print("average difference in tanhmoid with sqrt in tensorflow from fl16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d11),np.std(d11)))
    #print("average difference in tanhmoid with subtraction and sqrt in tensorflow from bl16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d5),np.std(d5)))

    #print("average difference from bf16 with f32 in bf16 is: {0}, std is: {1}\n".format(np.mean(d3),np.std(d3)))
    #print("average difference from fl16 with f32 is fl16 is: {0}, std is: {1}\n".format(np.mean(d2),np.std(d2)))

    
    
        
        
        
