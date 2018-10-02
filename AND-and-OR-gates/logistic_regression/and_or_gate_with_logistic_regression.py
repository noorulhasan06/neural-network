import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os, csv, shutil

# current program is of OR gate. For AND gate see below the comment.

learning_rate = 0.1
epoches = 500

def layer(x):
    w_init = tf.random_normal_initializer()   
    b_init = tf.constant_initializer(0)
    w = tf.get_variable(name='w', shape=[x.shape[1],1], initializer=w_init)
    b = tf.get_variable(name='b', shape=[1], initializer=b_init)
    z = tf.add(tf.matmul(x,w),b)
    return z

def loss(z,y):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = opt.minimize(cost, global_step=global_step)
    return train_op

def evaluate(z, y):
    correct = tf.equal(tf.round(tf.nn.sigmoid(z)), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

cost_val = []
if os.path.exists("logistic_logs/"):
    shutil.rmtree("logistic_logs/")
with tf.Graph().as_default():
    x = tf.placeholder(shape=[None, 2], dtype=tf.float32) #
    y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    z = layer(x)
    cost = loss(z, y)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = training(cost, global_step)
    accuracy = evaluate(z, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {x: np.array([[0,0],[0,1],[1,0],[1,1]])}
        print('current output',sess.run(tf.round(tf.nn.sigmoid(z)), feed_dict=feed_dict))
        feed_dict = {x: np.array([[0,0],[0,1],[1,0],[1,1]]), y: np.array([[0],[1],[1],[1]])} #for AND gate chage it with [[0],[0],[0],[1]]
        print(sess.run(accuracy, feed_dict=feed_dict))
        for i in range(epoches):
            sess.run(train_op, feed_dict=feed_dict)
            cost_val.append(sess.run(cost, feed_dict=feed_dict))
        print('accuracy',sess.run(accuracy, feed_dict=feed_dict))
        print('output',sess.run(tf.round(tf.nn.sigmoid(z)), feed_dict=feed_dict))
        saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
        plt.plot([i for i in range(len(cost_val))], cost_val)       #plot the error graph
        plt.show()
