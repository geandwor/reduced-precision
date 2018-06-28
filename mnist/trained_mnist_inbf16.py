#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:16:42 2018

@author: uic-cs
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf
#from tensorflow.python.client import timeline
outname="/home/uic-cs/Desktop/summerIntern/practicelearning/timeline1.json"

x = tf.placeholder(tf.bfloat16,[None,784])

W=tf.Variable(tf.zeros([784,10]))
W1 = tf.cast(W,tf.bfloat16)
b = tf.Variable(tf.zeros([10]))
b1 = tf.cast(b,tf.bfloat16)
sumweight = tf.matmul(x,W1)+b1
sumweight1 = tf.cast(sumweight,tf.float32)
y=tf.nn.softmax(sumweight1)
ybf16 = tf.cast(y,tf.bfloat16)
ybf16_fl32=tf.cast(ybf16,tf.float32)
y_ = tf.placeholder(tf.bfloat16,[None,10])
y_1 = tf.cast(y_,tf.float32)

cross_entropy = tf.reduce_mean(-1*tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

with tf.Session() as sess:
    profiler = tf.profiler.Profiler(sess.graph)
    
    tf.global_variables_initializer().run()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    
    #with open('timeline.json','w') as f:
        #tl=timeline.Timeline(run_metadata.step_stats)
    for i in range(1000):
        run_metadata = tf.RunMetadata()
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys},options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        #crf = tl.generate_chrome_trace_format()
        #f.write(crf)
        profiler.add_step(i,run_metadata)
        opts = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder().time_and_memory()).with_step(i).with_file_output(outname).build()
        #opts = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder().time_and_memory()).with_step(i).with_timeline_output(outname).build()

        profiler.profile_graph(options=opts)

    #profiler.advise()
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.bfloat16))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

"""weight initialization"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    #initial1 = tf.cast(initial,tf.bfloat16)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    #initial1 = tf.cast(initial,tf.bfloat16)
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
timer = []

xbf16 = tf.placeholder(tf.bfloat16,[None,784])
y_bf16 = tf.placeholder(tf.bfloat16,[None,10])

xbf16_fl32 = tf.cast(xbf16,tf.float32)
y_bf16_fl32 = tf.cast(y_bf16,tf.float32)

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(xbf16_fl32,[-1,28,28,1])



h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2X2(h_conv2)

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

"""readout layer"""
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_bf16_fl32, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_bf16_fl32,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  starter = time.time()
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        xbf16:batch[0], y_bf16: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    
  train_step.run(feed_dict={xbf16: batch[0], y_bf16: batch[1], keep_prob: 0.5})
  duration = time.time()-starter
  timer.append(duration)
print("test accuracy %g"%accuracy.eval(feed_dict={
    xbf16: mnist.test.images, y_bf16: mnist.test.labels, keep_prob: 1.0}))
    
sess.close()