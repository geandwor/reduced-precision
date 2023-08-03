#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:37:57 2018

@author: uic-cs

practice for tensorflow training for the MNIST data set
extract the parameters in each layers for the trained model
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf
from collections import defaultdict

#get corresponding index for nonzero element in one dimention array
def getIndex(array1):
    m = array1.shape[0]
    for i in range(m):
        if array1[i]>0.0:
            return i

#from tensorflow.python.client import timeline
outname="/home/uic-cs/Desktop/summerIntern/practicelearning/timeline1.json"

xbf16 = tf.placeholder(tf.bfloat16,[None,784])
xfl32 = tf.placeholder(tf.float32,[None,784])
xbf16_fl32 = tf.cast(xbf16,tf.float32)
xdif = tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(xfl32, xbf16_fl32)),
                                    reduction_indices=1))


Wfl32=tf.Variable(tf.zeros([784,10]))
Wbf = tf.Variable(tf.zeros([784,10]))
Wbf16 = tf.cast(Wbf, tf.bfloat16)
bfl32 = tf.Variable(tf.zeros([10]))
b = tf.Variable(tf.zeros([10]))
bbf16 = tf.cast(b,tf.bfloat16)
sumweightfl32 = tf.matmul(xfl32,Wfl32) +bfl32
sumweightbf16 = tf.matmul(xbf16,Wbf16)+bbf16
sumweightbf16_fl32 = tf.cast(sumweightbf16,tf.float32)
ybf16_fl32=tf.nn.softmax(sumweightbf16_fl32)
yfl32 = tf.nn.softmax(sumweightfl32)
Wbf16_fl32 = tf.cast(Wbf16, tf.float32)
bbf16_fl32 = tf.cast(bbf16, tf.float32)
#track the mean difference in parameters in bf16 and float32
Wdif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32,Wbf16_fl32))))
bdif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32,bbf16_fl32))))

#get the softmax difference between bf16 and fl32 with l2 norm
sumdif = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(yfl32,ybf16_fl32)),reduction_indices=[1]))

y_fl32 = tf.placeholder(tf.float32,[None,10])
y_bf16 = tf.placeholder(tf.bfloat16,[None,10])

y_bf16_fl32 = tf.cast(y_bf16,tf.float32)

rsumfl32 = tf.reduce_sum(y_fl32*tf.log(yfl32), reduction_indices=[1])
rsumbf16_fl32 = tf.reduce_sum(y_bf16_fl32*tf.log(ybf16_fl32), reduction_indices=[1])

#get the average difference of the ylogy between float 32 and bf16
rsumdif = tf.sqrt(tf.square(tf.subtract(rsumfl32,rsumbf16_fl32)))

cross_entropy_fl32 = tf.reduce_mean(-1*rsumfl32)

cross_entropy_bf16_fl32 = tf.reduce_mean(-1*rsumbf16_fl32)

centropydif = tf.subtract(cross_entropy_fl32,cross_entropy_bf16_fl32)

xdif = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xfl32, xbf16_fl32)), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(0.3)
vpairfl32 = optimizer.compute_gradients(cross_entropy_fl32)
trainfl32 = optimizer.apply_gradients(vpairfl32)
vpairbf16_fl32 = optimizer.compute_gradients(cross_entropy_bf16_fl32)
trainbf16_fl32 = optimizer.apply_gradients(vpairbf16_fl32)

rsumdiflist = []
sumdiflist = []
centropydiflist = []
Wdiflist = []
bdiflist = []
xdiflist = defaultdict(list)
xdic = defaultdict(list)

with tf.Session() as sess:
    profiler = tf.profiler.Profiler(sess.graph)
    
    tf.global_variables_initializer().run()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    
    #with open('timeline.json','w') as f:
        #tl=timeline.Timeline(run_metadata.step_stats)
    for i in range(20000):
        run_metadata = tf.RunMetadata()
        batch_xs, batch_ys = mnist.train.next_batch(100)
        #sess.run(trainfl32, feed_dict={xfl32:batch_xs, y_fl32:batch_ys},options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        #sess.run(trainbf16_fl32, feed_dict={xbf16:batch_xs, y_bf16:batch_ys},options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        sess.run(trainfl32, feed_dict={xfl32:batch_xs, y_fl32:batch_ys})
        sess.run(trainbf16_fl32, feed_dict={xbf16:batch_xs, y_bf16:batch_ys})
        for j in range(100):
            rsumdiflist.append(sess.run(rsumdif,feed_dict={xfl32:batch_xs, y_fl32:batch_ys,xbf16:batch_xs, y_bf16:batch_ys})[j])
            sumdiflist.append(sess.run(sumdif,feed_dict={xfl32:batch_xs, y_fl32:batch_ys,xbf16:batch_xs, y_bf16:batch_ys})[j])
        centropydiflist.append(sess.run(centropydif,feed_dict={xfl32:batch_xs, y_fl32:batch_ys,xbf16:batch_xs, y_bf16:batch_ys}))
        Wdiflist.append(sess.run(Wdif,feed_dict={xfl32:batch_xs, y_fl32:batch_ys,xbf16:batch_xs, y_bf16:batch_ys}))
        bdiflist.append(sess.run(bdif,feed_dict={xfl32:batch_xs, y_fl32:batch_ys,xbf16:batch_xs, y_bf16:batch_ys}))

        #crf = tl.generate_chrome_trace_format()
        #f.write(crf)
        #get profiling of operations' time
        #profiler.add_step(i,run_metadata)import matplotlib.pyplot as plt
        #opts = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder().time_and_memory()).with_step(i).with_file_output(outname).build()
        #opts = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder().time_and_memory()).with_step(i).with_timeline_output(outname).build()

        #profiler.profile_graph(options=opts)
    m,n = mnist.train.images.shape
    for i in range(m):
        index = getIndex(mnist.train.labels[i])
        xdiflist[index].append(sess.run(xdif,feed_dict={xfl32:mnist.train.images[i:i+1],xbf16:mnist.train.images[i:i+1]})[0])
        
    #profiler.advise()
    correct_predictionfl32 = tf.equal(tf.argmax(yfl32,1),tf.argmax(y_fl32,1))
    accuracyfl32 = tf.reduce_mean(tf.cast(correct_predictionfl32,tf.float32))
    print(sess.run(accuracyfl32, feed_dict={xfl32:mnist.test.images,y_fl32:mnist.test.labels}))
    print("Accuracy for fl32 is %g"%sess.run(accuracyfl32, feed_dict={xfl32:mnist.test.images,y_fl32:mnist.test.labels}))

    y_bf16_fl32 = tf.cast(y_bf16,tf.float32)
    correct_predictionbf16 = tf.equal(tf.argmax(ybf16_fl32,1),tf.argmax(y_bf16_fl32,1))
    correct_predictionfl32 = tf.equal(tf.argmax(yfl32,1),tf.argmax(y_fl32,1))
    accuracybf16 = tf.reduce_mean(tf.cast(correct_predictionbf16,tf.bfloat16))
    print("Accuracy for bf16 is %g"%sess.run(accuracybf16, feed_dict={xbf16:mnist.test.images,y_bf16:mnist.test.labels}))
m,n = mnist.train.images.shape
for i in range(m):
    index = getIndex(mnist.train.labels[i])
    for j in range(784):
        xdic[index].append(mnist.train.images[i][j])

import matplotlib.pyplot as plt

import numpy as np
def graphHist(dlist,name,label):
    plt.title("Histogram for "+label)
    #plt.hist(dlist,bins=20,normed=True)
    hist, bins = np.histogram(dlist)
    plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='blue')
    plt.xlabel("value in exponent 2")
    plt.gca().set_xscale('log',basex=2)
    #ind = np.arange(12)
    #plt.xticks(ind, ('-26','-24', '-20', '-17', '-14', '-10','-7','-4','-1','2','5','7'))
    plt.savefig("/home/uic-cs/Desktop/summerIntern/mnist/bf16mnistsimplenet/erroforlabel"+name+".png")
    plt.show()
    
for i in range(10):
    #plt.title("Histogram for label"+str(i))
    #plt.hist(xdiflist[i])
    #plt.xlabel("error in exponent 2")
    #plt.gca().set_xscale('log',basex=2)
    #plt.savefig("/home/uic-cs/Desktop/summerIntern/bf16mnistsimplenet/erroforlabel"+str(i)+"sigmoid.png")
    #plt.show()
    graphHist(xdic[i],"Distribute_forlabel "+str(i), "label "+str(i))


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

#construct network
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

xbf16 = tf.placeholder(tf.bfloat16,[None,784])
y_bf16 = tf.placeholder(tf.bfloat16,[None,10])

xfl32 = tf.placeholder(tf.float32,[None,784])
y_fl32= tf.placeholder(tf.float32,[None,10])

xbf16_fl32 = tf.cast(xbf16, tf.float32)
y_bf16_fl32 = tf.cast(y_bf16, tf.float32)

Wbf16_conv1 = weight_variable([5,5,1,32])
bbf16_conv1 = bias_variable([32])

Wfl32_conv1 = weight_variable([5,5,1,32])
bfl32_conv1 = bias_variable([32])
#get the mean difference in the parameters in the first convolution layer
W_conv1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32_conv1,Wbf16_conv1))))
b_conv1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32_conv1,bbf16_conv1))))


xfl32_image = tf.reshape(xfl32,[-1,28,28,1])
xbf16_fl32_image = tf.reshape(xbf16_fl32,[-1,28,28,1])

hfl32_conv1 = tf.nn.relu(conv2d(xfl32_image,Wfl32_conv1)+bfl32_conv1)
hfl32_pool1 = max_pool_2X2(hfl32_conv1)

hbf16_fl32_conv1 = tf.nn.relu(conv2d(xbf16_fl32_image,Wbf16_conv1)+bbf16_conv1)
hbf16_fl32_pool1 = max_pool_2X2(hbf16_fl32_conv1)




#get the average of the difference of the first pool layer
pool1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(hfl32_pool1,hbf16_fl32_pool1))))
Wfl32_conv2 = weight_variable([5,5,32,64])
bfl32_conv2 = bias_variable([64])

Wbf16_conv2 = weight_variable([5,5,32,64])
bbf16_conv2 = bias_variable([64])

#get the mean difference in the parameters in the second convolution layer
W_conv2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32_conv2,Wbf16_conv2))))
b_conv2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32_conv2,bbf16_conv2))))

hfl32_conv2 = tf.nn.relu(conv2d(hfl32_pool1,Wfl32_conv2)+bfl32_conv2)
hfl32_pool2 = max_pool_2X2(hfl32_conv2)

hbf16_fl32_conv2 = tf.nn.relu(conv2d(hbf16_fl32_pool1,Wbf16_conv2)+bbf16_conv2)
hbf16_fl32_pool2 = max_pool_2X2(hbf16_fl32_conv2)

pool2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(hfl32_pool2,hbf16_fl32_pool2))))

Wfl32_fc1 = weight_variable([7*7*64,1024])
bfl32_fc1 = bias_variable([1024])

Wbf16_fc1 = weight_variable([7*7*64,1024])
bbf16_fc1 = bias_variable([1024])

#get the mean difference in the parameters in the full connection layer
W_fc1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32_fc1,Wbf16_fc1))))
b_fc1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32_fc1,bbf16_fc1))))

hfl32_pool2_flat = tf.reshape(hfl32_pool2,[-1,7*7*64])
hfl32_fc1 = tf.nn.relu(tf.matmul(hfl32_pool2_flat,Wfl32_fc1)+bfl32_fc1)

hbf16_fl32_pool2_flat = tf.reshape(hbf16_fl32_pool2,[-1,7*7*64])
hbf16_fl32_fc1 = tf.nn.relu(tf.matmul(hbf16_fl32_pool2_flat,Wbf16_fc1)+bbf16_fc1)

fc1dif = pool1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(hfl32_fc1,hbf16_fl32_fc1))))


keepfl32_prob = tf.placeholder(tf.float32)
keepbf16_fl32_prob = tf.placeholder(tf.float32)

hfl32_fc1_drop = tf.nn.dropout(hfl32_fc1,keepfl32_prob)
hbf16_fl32_fc1_drop = tf.nn.dropout(hbf16_fl32_fc1,keepbf16_fl32_prob)

"""readout layer"""
Wfl32_fc2 = weight_variable([1024,10])
bfl32_fc2 = bias_variable([10])

Wbf16_fc2 = weight_variable([1024,10])
bbf16_fc2 = bias_variable([10])


#get the mean difference in the parameters in the full connection layer of readout layer
W_fc2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(Wfl32_fc2,Wbf16_fc2))))
b_fc2dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(bfl32_fc2,bbf16_fc2))))
yfl32_conv = tf.matmul(hfl32_fc1_drop,Wfl32_fc2)+bfl32_fc2
ybf16_fl32_conv = tf.matmul(hbf16_fl32_fc1_drop,Wbf16_fc2)+bbf16_fc2

yconvdif = pool1dif = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(yfl32_conv,ybf16_fl32_conv))))

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
  
  if i%100 == 0: 
    train_accuracyfl32 = accuracyfl32.eval(feed_dict={
        xfl32:batch[0], y_fl32: batch[1], keepfl32_prob: 1.0})
    print("step %d, training accuracy in fl32 %g"%(i, train_accuracyfl32))
     
  if i%100 == 0:
     train_accuracybf16_fl32 = accuracybf16_fl32.eval(feed_dict={xbf16:batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1.0})
     print("step %d, training accuracy in bf16 %g"%(i, train_accuracybf16_fl32))
     
    
  start_time = time.time()  
  train_stepfl32.run(feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 0.5})
  duration = time.time() - start_time
  timerfl32.append(duration)
  
  
  start_time = time.time()  
  train_stepbf16_fl32.run(feed_dict={xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 0.5})
  duration = time.time() - start_time
  timerbf16.append(duration)
  pool1diflist.append(sess.run(pool1dif,feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  pool2diflist.append(sess.run(pool2dif,feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  fc1diflist.append(sess.run(fc1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  yconvdiflist.append(sess.run(yconvdif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  centropydiflist.append(sess.run(centropydif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  W_conv1diflist.append(sess.run(W_conv1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  b_conv1diflist.append(sess.run(b_conv1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  W_conv2diflist.append(sess.run(W_conv2dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  b_conv2diflist.append(sess.run(b_conv2dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  W_fc1diflist.append(sess.run(W_fc1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  b_fc1diflist.append(sess.run(b_fc1dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  W_fc2diflist.append(sess.run(W_fc2dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))
  b_fc2diflist.append(sess.run(b_fc2dif, feed_dict={xfl32: batch[0], y_fl32: batch[1], keepfl32_prob: 1,xbf16: batch[0], y_bf16: batch[1], keepbf16_fl32_prob: 1}))

print("test accuracy in float32  %g"%accuracyfl32.eval(feed_dict={
    xfl32: mnist.test.images, y_fl32: mnist.test.labels, keepfl32_prob: 1.0}))
    
print("test accuracy in bfloat16  %g"%accuracybf16_fl32.eval(feed_dict={
    xbf16: mnist.test.images, y_bf16: mnist.test.labels, keepbf16_fl32_prob: 1.0}))
    







