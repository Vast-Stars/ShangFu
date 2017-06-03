# -*- coding: utf-8 -*-

import tensorflow as tf

# 定义神经网络参数
INPUT_NODE = 16384
OUTPUT_NODE = 2

IMAGE_SIZE = 128
NUM_CHANNELS = 3
NUM_LABELS = 2

# 第一层卷积层的深度和尺寸
CONV1_DEEP = 8
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 16
CONV2_SIZE = 5

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 32
CONV3_SIZE = 5

# 第四层卷积层的尺寸和深度
CONV4_DEEP = 64
CONV4_SIZE = 5

# 第五层卷积层的尺寸和深度
CONV5_DEEP = 128
CONV5_SIZE = 5

FC1_SIZE = 8192
FC2_SIZE = 512
FC3_SIZE = 10
# 定义前向传播的过程,这里添加了一个新的参数train，用于区分训练过程和测试过程
# 这个过程将用到dropout方法，dropout可以进一步提升模型可靠性并防止过拟合
# dropout只在训练时使用


def inference(input_tensor, train, regularizer):
    # 声明第一层卷积层的变量并实现前向传播过程
    # 通过使用不同的命名空间来隔离不同层的变量，这可以让每一层中的变量名只需要考虑在当前层中的作用，而不比担心重命名问题。
    # 这里定义的卷积层输入为128x128x3的图片。因为卷积层中使用了全零填充，所以输出为128x128x3x8的矩阵。

    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5, 深度为8的过滤器，过滤器移动步长为1, 并且使用全零填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程。这里选用最大池化层池化层过滤器的边长为2,
    # 使用全零填充且移动的步长为2.这一层的输入是上一层的输出，也就是124x124x3x8的矩阵。
    # 输出为64x64x3x8的矩阵。
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,  2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 声明第三层卷积层的变量并实现前向传播过程。这一层的输入为64x64x3x8的矩阵。
    # 输出为64x64x3x16
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为32的过滤器,过滤器步长为1,且使用全零填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程。这一层和前一层的结构是一样的。这一层的输入为64x64x3x16的矩阵。
    # 输出为32x32x3x16的矩阵
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 声明第5层卷积层的变量并实现前向传播过程。这一层的输入为32x32x3x16的矩阵。
    # 输出为32x32x3x32
    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable(
            "weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为32的过滤器,过滤器步长为1,且使用全零填充
        conv3 = tf.nn.conv2d(pool1, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    # 实现第6层池化层的前向传播过程。这一层和前一层的结构是一样的。这一层的输入为32x32x3x32的矩阵。
    # 输出为16x16x3x32的矩阵
    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 声明第7层卷积层的变量并实现前向传播过程。这一层的输入为32x32x3x32的矩阵。
    # 输出为32x32x3x64
    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable(
            "weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为64的过滤器,过滤器步长为1,且使用全零填充
        conv4 = tf.nn.conv2d(pool1, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    # 实现第8层池化层的前向传播过程。这一层和前一层的结构是一样的。这一层的输入为32x32x3x64的矩阵。
    # 输出为16x16x3x64的矩阵
    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 将第8层池化层的输出转化成第9层全连接层的输入格式。第8层的输出为16×16×3x64的矩阵。
        # 然而全连接层的输入格式为向量。在这里需要将这个16×16×3x32的矩阵拉直成一个向量
        pool_shape = pool4.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] * pool_shape[4]
        reshaped = tf.reshape(pool4, [pool_shape[0], nodes])

    # 声明第9层全连接层的变量并实现前向传播过程，这一层的输入是拉直以后的一组向量。
    # 向量长度为16x16x3x64=49152.输出是一组长度为512的向量。
    # 并引入了dropout的概念。dropout在训练时会随机将部分节点的输出改为0.dropout可以避免过你和问题，从而使得模型在测试数据上效果更好。
    # dropout一般只在全连接层而不是卷积层或者池化层使用。
    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC1_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化。
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC1_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [FC1_SIZE, FC2_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化。
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [FC2_SIZE], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(reshaped, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [FC2_SIZE, FC3_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化。
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [FC3_SIZE], initializer=tf.constant_initializer(0.1))

        fc3 = tf.nn.relu(tf.matmul(fc2, fc3_weights) + fc3_biases)
        if train: fc3 = tf.nn.dropout(fc3, 0.5)

    with tf.variable_scope('layer12-fc4'):
        fc4_weights = tf.get_variable("weight", [FC3_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc4_weights))
        fc4_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc3, fc4_weights) + fc4_biases

    return logit

