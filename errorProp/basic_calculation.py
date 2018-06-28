# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"basic matrix calculation
"basic matrix vector calculation
"basic convolution calculation

#matrix multiplication
import tensorflow as tf

mammal = tf.Variable("Elephant", tf.string)

#practice of variables

v = tf.Variable("v", shape=(), initializer=tf.zeros_initializer())
w = v+1
assignment = v.assign_add(1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(assignment)
sess.close()


hello = tf.constant("Hello, Tensorflow!")
sess = tf.Session()
print(sess.run(hello).decode('utf-8'))
sess.close()

node = tf.Variable(tf.zeros([2,2]))
with tf.Session() as sess:
    #initialize all gloabel variables
    sess.run(tf.global_variables_initializer())
    #evaluating node
    print("Tensor value before addition:\n",sess.run(node))
    
    #elementwise addition to tensor
    node = node.assign(node+tf.ones([2,2]))
    #evaluate node again
    print("Tensor value after addtion:\n", sess.run(node))

#generate random matrix
a = tf.Variable(tf.random_uniform([2,2],minval=50,maxval = 255,dtype = tf.float32), tf.float32)
b = tf.Variable(tf.random_uniform([2,2],minval = 100, maxval=255, dtype=tf.float32), tf.float32)

amatrix = tf.placeholder(tf.float16,(2,2))
bmatrix = tf.placeholder(tf.float16,(2,2))
amatrix0 = tf.placeholder(tf.bfloat16,(2,2))
bmatrix0 = tf.placeholder(tf.bfloat16,(2,2))
c0 = tf.matmul(amatrix0,bmatrix0)
c = tf.matmul(amatrix,bmatrix)
amatrix1 = tf.placeholder(tf.float32, (2,2))
bmatrix1 = tf.placeholder(tf.float32, (2,2))
c1 = tf.matmul(amatrix1,bmatrix1)
#scc1 = tf.subtract(c,c1)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        a = tf.random_uniform([2,2],minval=1e-3,maxval = 1e-2,dtype = tf.float32)
        b = tf.random_uniform([2,2],minval=1e-3,maxval = 1e-2,dtype = tf.float32)
        """a1 and b1 are in float32 format"""
        a1 = sess.run(a)
        b1 = sess.run(b)
        """down cast the a1, b1 before multiplication"""
        """down cast the a1, b1 after multiplication"""
        a1bf16 = sess.run(tf.cast(a1,tf.bfloat16))
        b1bf16 = sess.run(tf.cast(b1,tf.bfloat16))
        a1f16 = sess.run(tf.cast(a1,tf.float16))
        b1f16 = sess.run(tf.cast(b1,tf.float16))
        
        """calculate the multiplication"""
        """1 multiplication before down cast"""
        cc = sess.run(c, feed_dict= {amatrix:a1,bmatrix:b1})
        cc0 = sess.run(c0,feed_dict={amatrix0:a1,bmatrix0:b1})
        """upcast calcualted in 16 bit"""
        cccast = sess.run(tf.cast(cc,tf.float32))
        cc0cast = sess.run(tf.cast(cc0,tf.float32))
        """2 multiplication after down cast"""
        ccdown = sess.run(c,feed_dict = {amatrix:a1f16,bmatrix:b1f16})
        cc0down = sess.run(c0, feed_dict ={amatrix0:a1bf16,bmatrix0:b1bf16})
        ccdowncast = sess.run(tf.cast(ccdown,tf.float32))
        cc0downcast = sess.run(tf.cast(cc0down,tf.float32))
        
        cc1 = sess.run(c1, feed_dict= {amatrix1:a1,bmatrix1:b1})
        cc1_16 =sess.run(tf.cast(cc1,tf.float16))
        cc1_bf16 = sess.run(tf.cast(cc1, tf.bfloat16))
        """difference
        "1 difference for multiplication before downcast then result upcast"""
        diffcc0cast =cc1-cc0cast
        diffcccast = cc1 - cccast
        """"2 difference for multiplication after downcast then result upcast"""
        diffcc0downcast = cc1 - cc0downcast
        diffccdowncast = cc1 - ccdowncast
        """3 difference for multiplication result down cast to float16"""
        diffcc1_16 = cc1_16 - cc
        diffccdowncc1_16 = cc1_16 - ccdown
        """4 difference for multilplcation result down cast to bfloat16"""
        diffcc1_bf16 = cc1_bf16 - cc0
        diffcc0downcc1_bf16 = cc1_bf16 - cc0down
        
        
        print("epoch:{0} \n cmatrixbf16:{1} \n".format(i,cc0))
        print("epoch:{0} \n cmatrix16:{1}\n".format(i, cc))
        print("epoch:{0} \n cmatrixdownbf16:{1} \n".format(i,cc0down))
        print("epoch:{0} \n cmatrixdown16:{1}\n".format(i, ccdown))
        print("epoch:{0} \n cmatrix32:{1} \n".format(i,cc1))
        #difference before down cast then upcast
        print("#difference before down cast then upcast\n")
        print("epoch:{0} \n differencebf16before_down_then_up:{1}\n".format(i, diffcc0cast))
        print("epoch:{0} \n difference16before_down_then_up:{1} \n".format(i,diffcccast))
        
        #difference after downcast then upcast
        print("#difference after downcast then upcast")
        print("epoch:{0} \n differencebf16aftere_down_then_up:{1}\n".format(i, diffcc0downcast))
        print("epoch:{0} \n difference16bafter_down_then_up:{1} \n".format(i,diffccdowncast))
        
        #difference for mulitplication result down cast to float16
        print("epoch:{0} \n differencedown_16_before_down:{1}\n".format(i, diffcc1_16))
        print("epoch:{0} \n differencedown_16_after_down:{1} \n".format(i,diffccdowncc1_16))
        
        #difference for multiplication result down cast to bfloat16
        print("epoch:{0} \n differencedown_bf16_before_edown:{1}\n".format(i, diffcc1_bf16))
        print("epoch:{0} \n differencedown_bf16_after_down:{1} \n".format(i,diffcc0downcc1_bf16))


import tensorflow as tf
import numpy as np
mini = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1]
maxi = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1]
leng =len(mini)
shape1 = [2,2]
shape2 = [2,1]
ite = 100
for i in range(leng):
    calculateRange(mini[i],maxi[i],ite, shape1, shape2)
    
        
        
        
#generate random matrix
def calculateRange(mini,maxi,ite,shape1, shape2):
    a = tf.Variable(tf.random_uniform(shape1,minval=50,maxval = 255,dtype = tf.float32), tf.float32)
    b = tf.Variable(tf.random_uniform(shape2,minval = 100, maxval=255, dtype=tf.float32), tf.float32)

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
        d1 =float(0.0)
        d2 =float(0.0)
        d3 =float(0.0)
    
        for i in range(ite):
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
            d2 += np.sqrt(summation(diffcc1_bf16))
    print("--Iteration{0}\n".format(ite))
    print("--min {0} max {1}\n".format(mini[i],maxi[i]))
    print("average difference from bf16 with fl32 in fl32 is: {0}\n".format(d0/ite))
    print("average difference from fl16 with fl32 in fl32 is: {0}\n".format(d1/ite))
    print("average difference from bf16 with f32 in bf16 is: {0}\n".format(d3/ite))
    print("average difference from fl16 with f32 is fl16 is: {0}\n".format(d2/ite))

    
"""compare two generated random matrix """
"""
since based on the previos results of differences, the transformation of fl32 to fl16 or bfl16 would be the same
wherever the transformation happens before the feed or not
this will shorten the comparison lists
"""
import tensorflow as tf
import numpy as np
a = tf.Variable(tf.random_uniform([2,2],minval=50,maxval = 255,dtype = tf.float32), tf.float32)
b = tf.Variable(tf.random_uniform([2,2],minval = 100, maxval=255, dtype=tf.float32), tf.float32)

amatrix = tf.placeholder(tf.float16,[2,2])
bmatrix = tf.placeholder(tf.float16,[2,2])
amatrix0 = tf.placeholder(tf.bfloat16,[2,2])
bmatrix0 = tf.placeholder(tf.bfloat16,[2,2])
c0 = tf.matmul(amatrix0,bmatrix0)
c = tf.matmul(amatrix,bmatrix)
amatrix1 = tf.placeholder(tf.float32, [2,2])
bmatrix1 = tf.placeholder(tf.float32, [2,2])
c1 = tf.matmul(amatrix1,bmatrix1)
#scc1 = tf.subtract(c,c1)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    d0 =float(0.0)
    d1 =float(0.0)
    d2 =float(0.0)
    d3 =float(0.0)
    ite = 100
    for i in range(ite):
        a = tf.random_uniform([2,2],minval=180,maxval = 200,dtype = tf.float32)
        b = tf.random_uniform([2,2],minval=180,maxval = 200,dtype = tf.float32)
        """a1 and b1 are in float32 format"""
        a1 = sess.run(a)
        b1 = sess.run(b)
        """down cast the a1, b1 before multiplication"""
        """down cast the a1, b1 after multiplication"""
        #a1bf16 = sess.run(tf.cast(a1,tf.bfloat16))
        #b1bf16 = sess.run(tf.cast(b1,tf.bfloat16))
        #a1f16 = sess.run(tf.cast(a1,tf.float16))
        #b1f16 = sess.run(tf.cast(b1,tf.float16))
        
        """calculate the multiplication"""
        """1 multiplication before down cast"""
        cc = sess.run(c, feed_dict= {amatrix:a1,bmatrix:b1})
        cc0 = sess.run(c0,feed_dict={amatrix0:a1,bmatrix0:b1})
        """upcast calcualted in 16 bit"""
        cccast = sess.run(tf.cast(cc,tf.float32))
        cc0cast = sess.run(tf.cast(cc0,tf.float32))
        """2 multiplication after down cast"""
        #ccdown = sess.run(c,feed_dict = {amatrix:a1f16,bmatrix:b1f16})
        #cc0down = sess.run(c0, feed_dict ={amatrix0:a1bf16,bmatrix0:b1bf16})
        #ccdowncast = sess.run(tf.cast(ccdown,tf.float32))
        #cc0downcast = sess.run(tf.cast(cc0down,tf.float32))
        
        cc1 = sess.run(c1, feed_dict= {amatrix1:a1,bmatrix1:b1})
        cc1_16 =sess.run(tf.cast(cc1,tf.float16))
        cc1_bf16 = sess.run(tf.cast(cc1, tf.bfloat16))
        """difference
        "1 difference for multiplication before downcast then result upcast"""
        diffcc0cast =cc1-cc0cast
        diffcccast = cc1 - cccast
        #d0 = d0 + np.sqrt(diffcc0cast[0][0]**2+diffcc0cast[0][1]**2+diffcc0cast[1][0]**2+diffcc0cast[1][1]**2)
        #d1 = d1 + np.sqrt(diffcccast[0][0]**2+diffcccast[0][1]**2+diffcccast[1][0]**2+diffcccast[1][1]**2)

        """"2 difference for multiplication after downcast then result upcast"""
        #diffcc0downcast = cc1 - cc0downcast
        #diffccdowncast = cc1 - ccdowncast
        """3 difference for multiplication result down cast to float16"""
        diffcc1_16 = cc1_16 - cc
        #d2 = d2 + np.sqrt(diffcc1_16[0][0]**2+diffcc1_16[0][1]**2+diffcc1_16[1][0]**2+diffcc1_16[1][1]**2)

        #diffccdowncc1_16 = cc1_16 - ccdown
        """4 difference for multilplcation result down cast to bfloat16"""
        diffcc1_bf16 = cc1_bf16 - cc0
        #d3 = d3 + np.sqrt(diffcc1_bf16[0][0]**2+diffcc1_bf16[0][1]**2+diffcc1_bf16[1][0]**2+diffcc1_bf16[1][1]**2)
        
        
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
        
        """
        #diffcc0downcc1_bf16 = cc1_bf16 - cc0down
        print("epoch:{0} \n cmatrixbf16:{1} \n".format(i,cc0))
        print("epoch:{0} \n cmatrix16:{1}\n".format(i, cc))
        #print("epoch:{0} \n cmatrixdownbf16:{1} \n".format(i,cc0down))
        #print("epoch:{0} \n cmatrixdown16:{1}\n".format(i, ccdown))
        print("epoch:{0} \n cmatrix32:{1} \n".format(i,cc1))
        #difference before down cast then upcast
        print("#difference before down cast then upcast\n")
        print("epoch:{0} \n differencebf16before_down_then_up:{1}\n".format(i, diffcc0cast))
        print("epoch:{0} \n difference16before_down_then_up:{1} \n".format(i,diffcccast))
        
        #difference after downcast then upcast
#        print("#difference after downcast then upcast")
#        print("epoch:{0} \n differencebf16aftere_down_then_up:{1}\n".format(i, diffcc0downcast))
#        print("epoch:{0} \n difference16bafter_down_then_up:{1} \n".format(i,diffccdowncast))
        
        #difference for mulitplication result down cast to float16
        print("epoch:{0} \n differencedown_16_before_down:{1}\n".format(i, diffcc1_16))
        #print("epoch:{0} \n differencedown_16_after_down:{1} \n".format(i,diffccdowncc1_16))
        
        #difference for multiplication result down cast to bfloat16
        print("epoch:{0} \n differencedown_bf16_before_edown:{1}\n".format(i, diffcc1_bf16))
        #print("epoch:{0} \n differencedown_bf16_after_down:{1} \n".format(i,diffcc0downcc1_bf16))
    """
    print("average difference from bf16 with fl32 in fl32 is: {0}\n".format(d0/ite))
    print("average difference from fl16 with fl32 in fl32 is: {0}\n".format(d1/ite))
    print("average difference from bf16 with f32 in bf16 is: {0}\n".format(d3/ite))
    print("average difference from fl16 with f32 is fl16 is: {0}\n".format(d2/ite))


"""
matrix vector myltiplication in tensorflow
"""
import numpy as np
a = tf.Variable(tf.random_uniform([2,2],minval=50,maxval = 255,dtype = tf.float32), tf.float32)
b = tf.Variable(tf.random_uniform([2,1],minval = 100, maxval=255, dtype=tf.float32), tf.float32)

amatrix = tf.placeholder(tf.float16,(2,2))
bmatrix = tf.placeholder(tf.float16,(2,1))
amatrix0 = tf.placeholder(tf.bfloat16,(2,2))
bmatrix0 = tf.placeholder(tf.bfloat16,(2,1))
c0 = tf.matmul(amatrix0,bmatrix0)
c = tf.matmul(amatrix,bmatrix)
amatrix1 = tf.placeholder(tf.float32, (2,2))
bmatrix1 = tf.placeholder(tf.float32, (2,1))
c1 = tf.matmul(amatrix1,bmatrix1)
#scc1 = tf.subtract(c,c1)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    d0 =float(0.0)
    d1 =float(0.0)
    d2 =float(0.0)
    d3 =float(0.0)
    ite = 1000
    for i in range(ite):
        a = tf.random_uniform([2,2],minval=1e-3,maxval = 1e-2,dtype = tf.float32)
        b = tf.random_uniform([2,1],minval=1e-3,maxval = 1e-2,dtype = tf.float32)
        """a1 and b1 are in float32 format"""
        a1 = sess.run(a)
        b1 = sess.run(b)
        """down cast the a1, b1 before multiplication"""
        """down cast the a1, b1 after multiplication"""
        a1bf16 = sess.run(tf.cast(a1,tf.bfloat16))
        b1bf16 = sess.run(tf.cast(b1,tf.bfloat16))
        a1f16 = sess.run(tf.cast(a1,tf.float16))
        b1f16 = sess.run(tf.cast(b1,tf.float16))
        
        """calculate the multiplication"""
        """1 multiplication before down cast"""
        cc = sess.run(c, feed_dict= {amatrix:a1,bmatrix:b1})
        cc0 = sess.run(c0,feed_dict={amatrix0:a1,bmatrix0:b1})
        """upcast calcualted in 16 bit"""
        cccast = sess.run(tf.cast(cc,tf.float32))
        cc0cast = sess.run(tf.cast(cc0,tf.float32))
        """2 multiplication after down cast"""
        ccdown = sess.run(c,feed_dict = {amatrix:a1f16,bmatrix:b1f16})
        cc0down = sess.run(c0, feed_dict ={amatrix0:a1bf16,bmatrix0:b1bf16})
        ccdowncast = sess.run(tf.cast(ccdown,tf.float32))
        cc0downcast = sess.run(tf.cast(cc0down,tf.float32))
        
        cc1 = sess.run(c1, feed_dict= {amatrix1:a1,bmatrix1:b1})
        cc1_16 =sess.run(tf.cast(cc1,tf.float16))
        cc1_bf16 = sess.run(tf.cast(cc1, tf.bfloat16))
        """difference
        "1 difference for multiplication before downcast then result upcast"""
        diffcc0cast =cc1-cc0cast
        diffcccast = cc1 - cccast
        #d0 = d0 + np.sqrt(diffcc0cast[0][0]**2+diffcc0cast[0][1]**2+diffcc0cast[1][0]**2+diffcc0cast[1][1]**2)
        #d1 = d1 + np.sqrt(diffcccast[0][0]**2+diffcccast[0][1]**2+diffcccast[1][0]**2+diffcccast[1][1]**2)

        """"2 difference for multiplication after downcast then result upcast"""
        #diffcc0downcast = cc1 - cc0downcast
        #diffccdowncast = cc1 - ccdowncast
        """3 difference for multiplication result down cast to float16"""
        diffcc1_16 = cc1_16 - cc
        #d2 = d2 + np.sqrt(diffcc1_16[0][0]**2+diffcc1_16[0][1]**2+diffcc1_16[1][0]**2+diffcc1_16[1][1]**2)

        #diffccdowncc1_16 = cc1_16 - ccdown
        """4 difference for multilplcation result down cast to bfloat16"""
        diffcc1_bf16 = cc1_bf16 - cc0
        #d3 = d3 + np.sqrt(diffcc1_bf16[0][0]**2+diffcc1_bf16[0][1]**2+diffcc1_bf16[1][0]**2+diffcc1_bf16[1][1]**2)

        #diffcc0downcc1_bf16 = cc1_bf16 - cc0down
        
        
        print("epoch:{0} \n cmatrixbf16:{1} \n".format(i,cc0))
        print("epoch:{0} \n cmatrix16:{1}\n".format(i, cc))
        #print("epoch:{0} \n cmatrixdownbf16:{1} \n".format(i,cc0down))
        #print("epoch:{0} \n cmatrixdown16:{1}\n".format(i, ccdown))
        print("epoch:{0} \n cmatrix32:{1} \n".format(i,cc1))
        #difference before down cast then upcast
        print("#difference before down cast then upcast\n")
        print("epoch:{0} \n differencebf16before_down_then_up:{1}\n".format(i, diffcc0cast))
        print("epoch:{0} \n difference16before_down_then_up:{1} \n".format(i,diffcccast))
        
        #difference after downcast then upcast
#        print("#difference after downcast then upcast")
#        print("epoch:{0} \n differencebf16aftere_down_then_up:{1}\n".format(i, diffcc0downcast))
#        print("epoch:{0} \n difference16bafter_down_then_up:{1} \n".format(i,diffccdowncast))
        
        #difference for mulitplication result down cast to float16
        print("epoch:{0} \n differencedown_16_before_down:{1}\n".format(i, diffcc1_16))
        #print("epoch:{0} \n differencedown_16_after_down:{1} \n".format(i,diffccdowncc1_16))
        
        #difference for multiplication result down cast to bfloat16
        print("epoch:{0} \n differencedown_bf16_before_edown:{1}\n".format(i, diffcc1_bf16))
        #print("epoch:{0} \n differencedown_bf16_after_down:{1} \n".format(i,diffcc0downcc1_bf16))
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
        d2 += np.sqrt(summation(diffcc1_bf16))

    print("average difference from bf16 with fl32 in fl32 is: {0}\n".format(d0/ite))
    print("average difference from fl16 with fl32 in fl32 is: {0}\n".format(d1/ite))
    print("average difference from bf16 with f32 in bf16 is: {0}\n".format(d3/ite))
    print("average difference from fl16 with f32 is fl16 is: {0}\n".format(d2/ite))



#kernel, matrix  multiplication
"""
convolution involves matrix and kernel operation.
simple matrix 3*3
simple kernel 2*2
stride 1
no padding
"""
import numpy as np


amatrix = tf.placeholder(tf.float16,(3,3))
bmatrix = tf.placeholder(tf.float16,(2,2))
amatrix0 = tf.placeholder(tf.bfloat16,(3,3))
bmatrix0 = tf.placeholder(tf.bfloat16,(2,2))
c0 = tf.nn.conv1d(amatrix0,bmatrix0,1,'VALID')
c = tf.nn.conv1d(amatrix,bmatrix,1,'VAlID')
amatrix1 = tf.placeholder(tf.float32, (3,3))
bmatrix1 = tf.placeholder(tf.float32, (2,2))
c1 = tf.nn.conv1d(amatrix1,bmatrix1,'VALID')
#scc1 = tf.subtract(c,c1)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    d0 =float(0.0)
    d1 =float(0.0)
    d2 =float(0.0)
    d3 =float(0.0)
    ite = 100
    for i in range(ite):
        a = tf.random_uniform([2,2],minval=1e-4,maxval = 1e-3,dtype = tf.float32)
        b = tf.random_uniform([2,2],minval=1e-4,maxval = 1e-3,dtype = tf.float32)
        """a1 and b1 are in float32 format"""
        a1 = sess.run(a)
        b1 = sess.run(b)
        """down cast the a1, b1 before multiplication"""
        """down cast the a1, b1 after multiplication"""
        a1bf16 = sess.run(tf.cast(a1,tf.bfloat16))
        b1bf16 = sess.run(tf.cast(b1,tf.bfloat16))
        a1f16 = sess.run(tf.cast(a1,tf.float16))
        b1f16 = sess.run(tf.cast(b1,tf.float16))
        
        """calculate the multiplication"""
        """1 multiplication before down cast"""
        cc = sess.run(c, feed_dict= {amatrix:a1,bmatrix:b1})
        cc0 = sess.run(c0,feed_dict={amatrix0:a1,bmatrix0:b1})
        """upcast calcualted in 16 bit"""
        cccast = sess.run(tf.cast(cc,tf.float32))
        cc0cast = sess.run(tf.cast(cc0,tf.float32))
        """2 multiplication after down cast"""
        ccdown = sess.run(c,feed_dict = {amatrix:a1f16,bmatrix:b1f16})
        cc0down = sess.run(c0, feed_dict ={amatrix0:a1bf16,bmatrix0:b1bf16})
        ccdowncast = sess.run(tf.cast(ccdown,tf.float32))
        cc0downcast = sess.run(tf.cast(cc0down,tf.float32))
        
        cc1 = sess.run(c1, feed_dict= {amatrix1:a1,bmatrix1:b1})
        cc1_16 =sess.run(tf.cast(cc1,tf.float16))
        cc1_bf16 = sess.run(tf.cast(cc1, tf.bfloat16))
        """difference
        "1 difference for multiplication before downcast then result upcast"""
        diffcc0cast =cc1-cc0cast
        diffcccast = cc1 - cccast
        #d0 = d0 + np.sqrt(diffcc0cast[0][0]**2+diffcc0cast[0][1]**2+diffcc0cast[1][0]**2+diffcc0cast[1][1]**2)
        #d1 = d1 + np.sqrt(diffcccast[0][0]**2+diffcccast[0][1]**2+diffcccast[1][0]**2+diffcccast[1][1]**2)

        """"2 difference for multiplication after downcast then result upcast"""
        #diffcc0downcast = cc1 - cc0downcast
        #diffccdowncast = cc1 - ccdowncast
        """3 difference for multiplication result down cast to float16"""
        diffcc1_16 = cc1_16 - cc
        #d2 = d2 + np.sqrt(diffcc1_16[0][0]**2+diffcc1_16[0][1]**2+diffcc1_16[1][0]**2+diffcc1_16[1][1]**2)

        #diffccdowncc1_16 = cc1_16 - ccdown
        """4 difference for multilplcation result down cast to bfloat16"""
        diffcc1_bf16 = cc1_bf16 - cc0
        #d3 = d3 + np.sqrt(diffcc1_bf16[0][0]**2+diffcc1_bf16[0][1]**2+diffcc1_bf16[1][0]**2+diffcc1_bf16[1][1]**2)
        
        
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
        d2 += np.sqrt(summation(diffcc1_bf16))
        
        
        #diffcc0downcc1_bf16 = cc1_bf16 - cc0down
        print("epoch:{0} \n cmatrixbf16:{1} \n".format(i,cc0))
        print("epoch:{0} \n cmatrix16:{1}\n".format(i, cc))
        #print("epoch:{0} \n cmatrixdownbf16:{1} \n".format(i,cc0down))
        #print("epoch:{0} \n cmatrixdown16:{1}\n".format(i, ccdown))
        print("epoch:{0} \n cmatrix32:{1} \n".format(i,cc1))
        #difference before down cast then upcast
        print("#difference before down cast then upcast\n")
        print("epoch:{0} \n differencebf16before_down_then_up:{1}\n".format(i, diffcc0cast))
        print("epoch:{0} \n difference16before_down_then_up:{1} \n".format(i,diffcccast))
        
        #difference after downcast then upcast
#        print("#difference after downcast then upcast")
#        print("epoch:{0} \n differencebf16aftere_down_then_up:{1}\n".format(i, diffcc0downcast))
#        print("epoch:{0} \n difference16bafter_down_then_up:{1} \n".format(i,diffccdowncast))
        
        #difference for mulitplication result down cast to float16
        print("epoch:{0} \n differencedown_16_before_down:{1}\n".format(i, diffcc1_16))
        #print("epoch:{0} \n differencedown_16_after_down:{1} \n".format(i,diffccdowncc1_16))
        
        #difference for multiplication result down cast to bfloat16
        print("epoch:{0} \n differencedown_bf16_before_edown:{1}\n".format(i, diffcc1_bf16))
        #print("epoch:{0} \n differencedown_bf16_after_down:{1} \n".format(i,diffcc0downcc1_bf16))

    print("average difference from bf16 with fl32 in fl32 is: {0}\n".format(d0/ite))
    print("average difference from fl16 with fl32 in fl32 is: {0}\n".format(d1/ite))
    print("average difference from bf16 with f32 in bf16 is: {0}\n".format(d3/ite))
    print("average difference from fl16 with f32 is fl16 is: {0}\n".format(d2/ite))


        
#density of floating16 point
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

significand_bits = 10
exponent_min = -14
exponent_max = 15
binwidth = 0.001

fp_numbers = []
for exp in range(exponent_min, exponent_max+1):
    for sbits in range(0, 2**significand_bits):
        significand = 1 + sbits/2**significand_bits 
        fp_numbers.append(significand * 2**exp)
        
fp_numbers = np.array(fp_numbers)
pl.gca().set_xscale("log",basex=2)
pl.hist(fp_numbers, bins=np.logspace(-14,15, num=6000,base=2.0))

pl.show()

bins = 2.0**(np.arange(-14,-12))
plt.xscale('log',basex=2)
plt.hist(fp_numbers,bins = bins)
#print(fp_numbers)

#pt.hist(fp_numbers, bins=np.arange(min(fp_numbers),max(fp_numbers)+binwidth*10,binwidth))
pt.hist(fp_numbers, bins=5)
#pt.plot(fp_numbers, np.ones_like(fp_numbers), "+")
#pt.semilogx(fp_numbers, np.ones_like(fp_numbers), "+")

#density of bfloating 16

significand_bits = 7
exponent_min = -254
exponent_max = 255

fp_numbers = []
for exp in range(exponent_min, exponent_max+1):
    for sbits in range(0, 2**significand_bits):
        significand = 1 + sbits/2**significand_bits 
        fp_numbers.append(significand * 2**exp)
        
fp_numbers = np.array(fp_numbers)
pt.hist(fp_numbers, bins=3)
bins = 2.0**(np.arange(-14,-12))
plt.xscale('log',basex=2)
plt.hist(fp_numbers,bins = bins)
        



#creating nodes in computation graph
a = tf.placeholder(tf.int32, shape=(3,1))
b = tf.placeholder(tf.int32, shape=(1,3))
c = tf.matmul(a,b)

#running computation graph
with tf.Session() as sess:
    print(sess.run(c,feed_dict={a:[[3],[2],[1]], b:[[1,2,3]]}))
    print(sess.run(a))
    print(sess.run(b))
    
    
    

a = tf.constant([1,2,3,4,5,6], shape=[2,3])
b = tf.constant([7,8,9,10,11,12], shape=[3,2])

c = tf.matmul(a,b)

#convolution

i = tf.constant([1, 0, 2, 3, 0, 1, 1], dtype=tf.float32, name='i')
k = tf.constant([2, 1, 3], dtype=tf.float32, name='k')

print (i, '\n', k, '\n')

data   = tf.reshape(i, [1, int(i.shape[0]), 1], name='data')
kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name='kernel')

print (data, '\n', kernel, '\n')

res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, 'VALID'))
with tf.Session() as sess:
    print(sess.run(res))
    
    
#convolution with padding
res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, 'SAME'))
with tf.Session() as sess:
    print (sess.run(res))
    
#convolution with sriding
res = tf.squeeze(tf.nn.conv1d(data, kernel, 2, 'SAME'))
with tf.Session() as sess:
    print (sess.run(res))    