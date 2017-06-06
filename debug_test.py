from read_TFrecord import read_and_decode
import tensorflow as tf
from Generate_TFcorder import dense_to_one_hot

images, labels = read_and_decode("test.tf")
xs, ys = tf.train.shuffle_batch([images, labels], batch_size=5, capacity=15, min_after_dequeue=10)
yss = tf.one_hot(ys, 2)
with tf.Session() as sess:
    print('Session Begin!')
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    xs_test, ys_test,yss_test = sess.run([xs, ys,yss])
    print(ys_test)
    print(yss_test)
