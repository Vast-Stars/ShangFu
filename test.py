from read_TFrecord import read_and_decode
import tensorflow as tf

images, labels = read_and_decode("test2.tf")
xs, ys = tf.train.shuffle_batch([images, labels], batch_size=5, capacity=15, min_after_dequeue=10)
with tf.Session() as sess:
    print('Session Begin!')
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    xs_test, ys_test = sess.run([xs, ys])
    print(ys_test)
