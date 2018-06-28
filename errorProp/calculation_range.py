#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:31:54 2018

@author: uic-cs

compare two randomized generated input for designed shape to get the range
"""

import tensorflow as tf
import numpy as np

#generate random matrix
def calculateRange(mini,maxi,ite,shape1, shape2):
    
    

    amatrix = tf.placeholder(tf.float16,shape1)
    bmatrix = tf.placeholder(tf.float16,shape2)
    amatrix0 = tf.placeholder(tf.bfloat16,shape1)
    bmatrix0 = tf.placeholder(tf.bfloat16,shape2)
    c0 = tf.matmul(amatrix0,bmatrix0)
    c = tf.matmul(amatrix,bmatrix)
    amatrix1 = tf.placeholder(tf.float32, shape1)
    bmatrix1 = tf.placeholder(tf.float32, shape2)
    c1 = tf.matmul(amatrix1,bmatrix1)
    #scc1 = tf.subtract(c,c1)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        d0 =float(0.0)
        d1 =float(0.0)#python data types floating
        d2 =float(0.0)
        d3 =float(0.0)
    
        for j in range(ite):
            a = tf.random_uniform(shape1,minval=mini,maxval = maxi,dtype = tf.float32)
            b = tf.random_uniform(shape2,minval=mini,maxval = maxi,dtype = tf.float32)
            """a1 and b1 are in float32 format"""
            a1 = sess.run(a)
            b1 = sess.run(b)
          
            
            """calculate the multiplication"""
            """1 multiplication before down cast"""
            cc = sess.run(c, feed_dict= {amatrix:a1,bmatrix:b1})
            cc0 = sess.run(c0,feed_dict={amatrix0:a1,bmatrix0:b1})
            """upcast calcualted in 16 bit"""
            cccast = sess.run(tf.cast(cc,tf.float32))
            cc0cast = sess.run(tf.cast(cc0,tf.float32))
     
        
            cc1 = sess.run(c1, feed_dict= {amatrix1:a1,bmatrix1:b1})
            cc1_16 =sess.run(tf.cast(cc1,tf.float16))
            cc1_bf16 = sess.run(tf.cast(cc1, tf.bfloat16))
            """difference
            "1 difference for multiplication before downcast then result upcast"""
            diffcc0cast =cc1-cc0cast
            diffcccast = cc1 - cccast

            """"2 difference for multiplication after downcast then result upcast"""
            """3 difference for multiplication result down cast to float16"""
            diffcc1_16 = cc1_16 - cc

            """4 difference for multilplcation result down cast to bfloat16"""
            diffcc1_bf16 = cc1_bf16 - cc0

        
            """summation of the elements of the difference matrix and get the square root"""
            #assume the List is either vector or two dimention matrix
            def summation(ndarray):
                s = float(0)
                m,n = ndarray.shape
                for i in range(m):
                    for j in range(n):
                        s +=ndarray[i][j]**2
                  
                return s
       
            d0 += np.sqrt(summation(diffcc0cast))
            d1 += np.sqrt(summation(diffcccast))
            d2 += np.sqrt(summation(diffcc1_16))
            d3 += np.sqrt(summation(diffcc1_bf16))
    print("--Iteration {0}\n".format(ite))
    print("--min {0} max {1}\n".format(mini,maxi))
    print("average difference from bf16 with fl32 in fl32 is: {0}\n".format(d0/ite))
    print("average difference from fl16 with fl32 in fl32 is: {0}\n".format(d1/ite))
    print("average difference from bf16 with f32 in bf16 is: {0}\n".format(d3/ite))
    print("average difference from fl16 with f32 is fl16 is: {0}\n".format(d2/ite))

import tensorflow as tf
import numpy as np

mini = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,100]
maxi = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,100,180]
leng =len(mini)
shape1 = [2,2]
shape2 = [2,2]
ite = 100
"""add code here"""
#a = tf.Variable(tf.random_uniform(shape1,minval=50,maxval = 255,dtype = tf.float32), tf.float32)
#b = tf.Variable(tf.random_uniform(shape2,minval = 100, maxval=255, dtype=tf.float32), tf.float32)

amatrix = tf.placeholder(tf.float16,shape1)
bmatrix = tf.placeholder(tf.float16,shape2)
amatrix0 = tf.placeholder(tf.bfloat16,shape1)
bmatrix0 = tf.placeholder(tf.bfloat16,shape2)
c0 = tf.matmul(amatrix0,bmatrix0)
c = tf.matmul(amatrix,bmatrix)
amatrix1 = tf.placeholder(tf.float32, shape1)
bmatrix1 = tf.placeholder(tf.float32, shape2)
c1 = tf.matmul(amatrix1,bmatrix1)
#scc1 = tf.subtract(c,c1)
init = tf.global_variables_initializer()
for i in range(leng):
    print("iteration i{0}\n".format(i))
    with tf.Session() as sess:
        sess.run(init)
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
            """upcast calcualted in 16 bit"""
            cccast = sess.run(tf.cast(cc,tf.float32))
            cc0cast = sess.run(tf.cast(cc0,tf.float32))
     
        
            cc1 = sess.run(c1, feed_dict= {amatrix1:a1,bmatrix1:b1})
            cc1_16 =sess.run(tf.cast(cc1,tf.float16))
            cc1_bf16 = sess.run(tf.cast(cc1, tf.bfloat16))
            """difference
            "1 difference for multiplication before downcast then result upcast"""
            diffcc0cast =cc1-cc0cast
            #subtract in tensorflow in float32
            diffcc0castf=sess.run(tf.subtract(cc1,cc0cast))
            diffcccast = cc1 - cccast
            #subtract in tensorflow in float32
            diffcccastf=sess.run(tf.subtract(cc1,cccast))

            """"2 difference for multiplication after downcast then result upcast"""
            """3 difference for multiplication result down cast to float16"""
            diffcc1_16 = cc1_16 - cc

            """4 difference for multilplcation result down cast to bfloat16"""
            diffcc1_bf16 = cc1_bf16 - cc0

        
            """summation of the elements of the difference matrix and get the square root"""
            #assume the List is either vector or two dimention matrix
            def summation(ndarray):
                s = float(0)
                m,n = ndarray.shape
                for i in range(m):
                    for j in range(n):
                        s +=ndarray[i][j]**2
                  
                return s
       
            d0.append(np.sqrt(summation(diffcc0cast)))
            #square root in tensorflow in float32
            d01.append(sess.run(tf.sqrt(summation(diffcc0cast))))
            d1.append(np.sqrt(summation(diffcccast)))
            #square root in tensorflow in float32
            d11.append(sess.run(tf.sqrt(summation(diffcccast))))
            d2.append(np.sqrt(summation(diffcc1_16)))
            d3.append(np.sqrt(summation(diffcc1_bf16)))
            #square root in tensorflow in float32
            d4.append(sess.run(tf.sqrt(summation(diffcc0castf))))
            #square root in tensorflow in float32
            d5.append(sess.run(tf.sqrt(summation(diffcccastf))))
    print("--Iteration {0}\n".format(ite))
    print("--min {0} max {1}\n".format(mini[i],maxi[i]))
    print("average difference from bf16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d0),np.std(d0)))
    print("average difference with sqrt in tensorflow from bf16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d01),np.std(d01)))
    print("average difference with subtraction and sqrt in tensorflow from bf16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d4),np.std(d4)))

    print("average difference from fl16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d1),np.std(d1)))
    print("average difference with sqrt in tensorflow from fl16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d11),np.std(d11)))
    print("average difference with subtraction and sqrt in tensorflow from bl16 with fl32 in fl32 is: {0}, std is: {1}\n".format(np.mean(d5),np.std(d5)))

    #print("average difference from bf16 with f32 in bf16 is: {0}, std is: {1}\n".format(np.mean(d3),np.std(d3)))
    #print("average difference from fl16 with f32 is fl16 is: {0}, std is: {1}\n".format(np.mean(d2),np.std(d2)))

    
    
        
        
        
