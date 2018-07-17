#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:13:14 2018

@author: uic-cs


Convoluted neural net:
two convoluted neural nets
two max pooling layers
two fully connected layers
activation function relu

for bf16:
calculations is done in bf16   
all the parameters used with bf16 are declared in bfl16 except the first convoluted layers in fl32 and the first  conv layers in fl32
matmul,  maxpool, relu in second layers an full connected layers
calculations done in fl32
relu in 1st convoluted layers,conv2d, nn.dropout, softmax cross entropy




"""


"""weight initialization"""
def weight_variablebf16(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.constant(0.02,shape=shape) # starting from the same set of initial values
    initial1 = tf.cast(initial,tf.bfloat16)
    return tf.Variable(initial1)
def bias_variablebf16(shape):
    initial = tf.constant(0.1,shape=shape)
    initial1 = tf.cast(initial,tf.bfloat16)
    return tf.Variable(initial1)





"""weight initialization"""
def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.constant(0.02,shape=shape) # starting from the same set of initial values
    #initial1 = tf.cast(initial,tf.float16)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    #initial1 = tf.cast(initial,tf.float16)
    return tf.Variable(initial)

"""convolution and pooling bf16 to fl32 since in 1.8.0, conv2d doesn't support bfloat16"""
def conv2dbf16(x,W):
    Wfl32 = tf.cast(W,tf.float32)
    xfl32 = tf.cast(x,tf.float32)
    return tf.nn.conv2d(xfl32,Wfl32,strides=[1,1,1,1],padding='SAME')

#def max_pool_2X2(x):
#    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

"""convolution and pooling"""
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf
import time
timerfl32 = []
timerbf16 = []

pool1diflist = []
pool2diflist = []
fc1diflist = []
yconvdiflist = []
centropydiflist = []
W_conv1diflist = []
b_conv1diflist = []
W_conv2diflist = []
b_conv2diflist = []
W_fc1diflist = []
b_fc1diflist = []
W_fc2diflist = []
b_fc2diflist = []
bf16resultlist =[]
fl32resultlist = []

xbf16 = tf.placeholder(tf.bfloat16,[None,784])
y_bf16 = tf.placeholder(tf.bfloat16,[None,10])

xfl32 = tf.placeholder(tf.float32,[None,784])
y_fl32= tf.placeholder(tf.float32,[None,10])

xbf16_fl32 = tf.cast(xbf16, tf.float32)
y_bf16_fl32 = tf.cast(y_bf16, tf.float32)

Wbf16_fl32conv1 = weight_variable([5,5,1,32])
bbf16_fl32conv1 = bias_variable([32])


Wbf16_conv1 = tf.cast(Wbf16_fl32conv1,tf.bfloat16)
bbf16_conv1 = tf.cast(bbf16_fl32conv1,tf.bfloat16)


Wfl32_conv1 = weight_variable([5,5,1,32])
bfl32_conv1 = bias_variable([32])
#get the mean difference in the parameters in the first convolution layer
#W_conv1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32_conv1,Wbf16_conv1))))
#b_conv1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32_conv1,bbf16_conv1))))


xfl32_image = tf.reshape(xfl32,[-1,28,28,1])
xbf16_image = tf.reshape(xbf16,[-1,28,28,1])
xbf16_fl32_image = tf.cast(xbf16_image,tf.float32)

hfl32_conv1 = tf.nn.relu(conv2d(xfl32_image,Wfl32_conv1)+bfl32_conv1)
hfl32_pool1 = max_pool_2X2(hfl32_conv1)

#here, tf.nn.conv2d doen't support bfloat16
conv1bf16_fl32 = conv2d(xbf16_fl32_image,Wbf16_fl32conv1)
conv1bf16 = tf.cast(conv1bf16_fl32, tf.bfloat16)
bbf16_fl32_conv1 = tf.cast(bbf16_conv1,tf.float32)
hbf16_fl32_conv1 = tf.nn.relu(conv1bf16_fl32+bbf16_fl32_conv1)
hbf16_conv1 = tf.cast(hbf16_fl32_conv1,tf.bfloat16)
hbf16_pool1 = max_pool_2X2(hbf16_conv1)



#get the average of the difference of the first pool layer
#pool1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(hfl32_pool1,hbf16_fl32_pool1))))


Wfl32_conv2 = weight_variable([5,5,32,64])
bfl32_conv2 = bias_variable([64])

Wbf16_conv2 = weight_variablebf16([5,5,32,64])
bbf16_conv2 = bias_variablebf16([64])

#get the mean difference in the parameters in the second convolution layer
#W_conv2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32_conv2,Wbf16_conv2))))
#b_conv2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32_conv2,bbf16_conv2))))

hfl32_conv2 = tf.nn.relu(conv2d(hfl32_pool1,Wfl32_conv2)+bfl32_conv2)
hfl32_pool2 = max_pool_2X2(hfl32_conv2)

#here, tf.nn.conv2d doen't support bfloat16
conv2bf16_fl32 = conv2dbf16(hbf16_pool1,Wbf16_conv2)
conv2bf16 = tf.cast(conv2bf16_fl32, tf.bfloat16)
#bbf16_fl32_conv2 = tf.cast(bbf16_conv2, tf.float32)
hbf16_conv2 = tf.nn.relu(conv2bf16+bbf16_conv2)
hbf16_conv2 = tf.cast(hbf16_conv2,tf.bfloat16)
hbf16_pool2 = max_pool_2X2(hbf16_conv2)

#pool2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(hfl32_pool2,hbf16_fl32_pool2))))

Wfl32_fc1 = weight_variable([7*7*64,1024])
bfl32_fc1 = bias_variable([1024])

Wbf16_fc1 = weight_variablebf16([7*7*64,1024])
bbf16_fc1 = bias_variablebf16([1024])

#get the mean difference in the parameters in the full connection layer
#W_fc1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32_fc1,Wbf16_fc1))))
#b_fc1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32_fc1,bbf16_fc1))))

hfl32_pool2_flat = tf.reshape(hfl32_pool2,[-1,7*7*64])
hfl32_fc1 = tf.nn.relu(tf.matmul(hfl32_pool2_flat,Wfl32_fc1)+bfl32_fc1)

hbf16_pool2_flat = tf.reshape(hbf16_pool2,[-1,7*7*64])
hbf16_fc1 = tf.nn.relu(tf.matmul(hbf16_pool2_flat,Wbf16_fc1)+bbf16_fc1)

#fc1dif = pool1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(hfl32_fc1,hbf16_fl32_fc1))))


keepfl32_prob = tf.placeholder(tf.float32)
keepbf16_prob = tf.placeholder(tf.float32)

hfl32_fc1_drop = tf.nn.dropout(hfl32_fc1,keepfl32_prob)

hbf16_fc1_fl32 = tf.cast(hbf16_fc1,tf.float32)

hbf16_fc1_drop_fl32 = tf.nn.dropout(hbf16_fc1_fl32,keepbf16_prob)

"""readout layer"""
Wfl32_fc2 = weight_variable([1024,10])
bfl32_fc2 = bias_variable([10])

Wbf16_fc2 = weight_variablebf16([1024,10])
bbf16_fc2 = bias_variablebf16([10])


#get the mean difference in the parameters in the full connection layer of readout layer
#W_fc2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32_fc2,Wbf16_fc2))))
#b_fc2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32_fc2,bbf16_fc2))))
yfl32_conv = tf.matmul(hfl32_fc1_drop,Wfl32_fc2)+bfl32_fc2
hbf16_fc1_drop = tf.cast(hbf16_fc1_drop_fl32,tf.bfloat16)
ybf16_conv = tf.matmul(hbf16_fc1_drop,Wbf16_fc2)+bbf16_fc2

#yconvdif = pool1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(yfl32_conv,ybf16_fl32_conv))))
ybf16_fl32_conv = tf.cast(ybf16_conv, tf.float32)

cross_entropyfl32 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_fl32, logits=yfl32_conv))
cross_entropybf16_fl32 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_bf16_fl32, logits=ybf16_fl32_conv))
centropydif = pool1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(cross_entropyfl32,cross_entropybf16_fl32))))

train_stepfl32 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropyfl32)
train_stepbf16_fl32 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropybf16_fl32)

correct_predictionfl32 = tf.equal(tf.argmax(yfl32_conv,1), tf.argmax(y_fl32,1))
accuracyfl32 = tf.reduce_mean(tf.cast(correct_predictionfl32, tf.float32))
correct_predictionbf16_fl32 = tf.equal(tf.argmax(ybf16_fl32_conv,1), tf.argmax(y_bf16_fl32,1))
accuracybf16_fl32 = tf.reduce_mean(tf.cast(correct_predictionbf16_fl32, tf.float32))

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
for i in range(25000):
  batch = mnist.train.next_batch(50)
  
  if i%1000 == 0:
     
    train_accuracyfl32 = accuracyfl32.eval(feed_dict={
        xfl32: mnist.test.images, y_fl32: mnist.test.labels, keepfl32_prob: 1.0})
    fl32resultlist.append(train_accuracyfl32)
    print("step %d, training accuracy for testing data in fl32 %g"%(i, train_accuracyfl32))
    
    
  
  if i%1000 == 0:
     train_accuracybf16_fl32 = accuracybf16_fl32.eval(feed_dict={xbf16: mnist.test.images, y_bf16: mnist.test.labels, keepbf16_prob: 1.0})
     bf16resultlist.append(train_accuracybf16_fl32)
     print("step %d, training accuracy for testing data in bf16 %g"%(i, train_accuracybf16_fl32))
     
    
  start_time = time.time()  
  train_stepfl32.run(feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 0.5})
  duration = time.time() - start_time
  timerfl32.append(duration)
  
  
  start_time = time.time()  
  train_stepbf16_fl32.run(feed_dict={xbf16: batch[0], y_bf16: batch[1], keepbf16_prob: 0.5})
  duration = time.time() - start_time
  timerbf16.append(duration)
#  pool1diflist.append(sess.run(pool1dif,feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  pool2diflist.append(sess.run(pool2dif,feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  fc1diflist.append(sess.run(fc1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  yconvdiflist.append(sess.run(yconvdif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  centropydiflist.append(sess.run(centropydif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  W_conv1diflist.append(sess.run(W_conv1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  b_conv1diflist.append(sess.run(b_conv1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  W_conv2diflist.append(sess.run(W_conv2dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  b_conv2diflist.append(sess.run(b_conv2dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  W_fc1diflist.append(sess.run(W_fc1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  b_fc1diflist.append(sess.run(b_fc1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  W_fc2diflist.append(sess.run(W_fc2dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
#  b_fc2diflist.append(sess.run(b_fc2dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))

print("test accuracy in float32  %g"%accuracyfl32.eval(feed_dict={
    xfl32: mnist.test.images, y_fl32: mnist.test.labels, keepfl32_prob: 1.0}))
    
print("test accuracy in bfloat16  %g"%accuracybf16_fl32.eval(feed_dict={
    xbf16: mnist.test.images, y_bf16: mnist.test.labels, keepbf16_prob: 1.0}))
    


#draw bar plot 
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(len(bf16resultlist))

fig,ax = plt.subplots()
plt.plot(x,fl32resultlist,marker='.', linestyle='--', color='r',label = "float32")
plt.plot(x,bf16resultlist,marker= '.',linestyle = '--',color = 'b',label = 'bfloat16')
plt.xticks(rotation=70)
#plt.xticks(x)
plt.title("Accuracy over training for Fl32 and Bf16 with cross_entropy")
plt.xlabel("Number of Iterations in thousands")
plt.ylabel("Batch Accuracy")
plt.legend()
plt.savefig("/home/uic-cs/Desktop/summerIntern/accuracy_fl32bf16convoluted_more_iterationssetup1_2.png")
plt.show()




