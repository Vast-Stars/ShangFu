import tensorflow as tf
import tf_inference
# 定义神经网络相关的参数
import Variables
# 定义训练过程
import os

def train(x, y_,
          drop_out=True,
           MODEL_SAVE_PATH=os.getcwd()+"/SAVE/model.ckpt",
           BATCH_SIZE=Variables.BATCH_SIZE,
          IMAGE_SIZE=Variables.IMAGE_SIZE,
           NUM_CHANNELS=Variables.NUM_CHANNELS):

    regularizer = tf.contrib.layers.l2_regularizer(Variables.REGULARIZATION_RATE)
    # 直接使用tf.inference.py中的前向传播过程
    y = tf_inference.inference(x,drop_out= drop_out,regularizer=regularizer)
    # 将代表训练轮数的变量设为不可训练的参数。
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平类
    # 给定训练轮数的变量可以加快早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(Variables.MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均。
    # 其他滑动变量(比如global_step)就不需要了。
    # tf.trainable_variables返回的就是图上的集合 GraphKeys.TRAINABLE_VARIABLES 中的元素。
    # 这个集合中的元素就是所有没有指定trainable = False 的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算交叉熵作为刻画预测直和实际直之间差距的损失函数。这里使用了sparse_softmax_cross_entropy_with_logits函数
    # 来计算交叉熵。当分类问题只有一个正确答案时，可以使用这个函数加速交叉熵的计算。
    # 这个函数的第一个参数是神经网络不包含softmax层的前向传播结果，第二个是训练数据的正确答案。
    # 标准答案是一个长度为2的二位数组。该函数需要提供的是一个正确答案的数字，所以要使用tf.argmax函数来得到正确答案对应的类别编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, dimension=1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    with tf.name_scope('loss'):
        loss = cross_entropy_mean
        if drop_out==True:
            loss+=tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)
    learning_rate = tf.train.exponential_decay(
                            learning_rate=Variables.LEARNING_RATE_BASE,
                            global_step=global_step,
                            decay_steps=60000 / BATCH_SIZE, decay_rate= Variables.LEARNING_RATE_DECAY,
                            staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver(tf.trainable_variables())

    # tf.scalar_summary("cost_function", loss)
    # merged_summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        print('Session Begin!')

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter("logs/", sess.graph)

        for i in range(Variables.TRAINING_STEPS):
            _, loss_value, tem_y= sess.run([train_op, loss, y_])

            # DEBUG用。输出一些训练信息  第三项是样本混合均匀度，比较理想是在65—75之间
            print( '%-2s: %-10s,%s' % (i, loss_value, tem_y.cumsum(0)[BATCH_SIZE-1][0]))
            rs = sess.run(merged)

            if i % 50== 0:
                writer.add_summary(rs, i)
                #print("After %d training step(s), loss on training batch is %g." % (i, loss_value))
                #saver.save(sess, MODEL_SAVE_PATH)
        coord.request_stop()
        coord.join(threads)
