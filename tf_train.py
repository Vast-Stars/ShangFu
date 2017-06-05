import tensorflow as tf

import tf_inference
import os
import numpy as np
import read_TFrecord
# 定义神经网络相关的参数


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "/model/model.ckpt"
#MODEL_NAME = "model.ckpt"


# 定义训练过程
def train(images, labels, sess):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            tf_inference.IMAGE_SIZE,
            tf_inference.IMAGE_SIZE,
            tf_inference.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, tf_inference.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用tf.inference.py中的前向传播过程
    y = tf_inference.inference(x, False, regularizer)
    # 将代表训练轮数的变量设为不可训练的参数。
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平类
    # 给定训练轮数的变量可以加快早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均。
    # 其他滑动变量(比如global_step)就不需要了。
    # tf.trainable_variables返回的就是图上的集合 GraphKeys.TRAINABLE_VARIABLES 中的元素。
    # 这个集合中的元素就是所有没有指定trainable = False 的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算交叉熵作为刻画预测直和实际直之间差距的损失函数。这里使用了sparse_softmax_cross_entropy_with_logits函数
    # 来计算交叉熵。当分类问题只有一个正确答案时，可以使用这个函数加速交叉熵的计算。
    # 这个函数的第一个参数是神经网络不包含softmax层的前向传播结果，第二个是训练数据的正确答案。
    # 标准答案是一个长度为2的二位数组。该函数需要提供的是一个正确答案的数字，所以要使用tf.argmax函数来得到正确答案对应的类别编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
        xs, ys = tf.train.shuffle_batch([images, labels], 1000, 4000, 1000)

        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
        saver.save(sess, MODEL_SAVE_PATH)

        if i % 1000 == 0:
            print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
