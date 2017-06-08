from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tf_train import train
from read_TFrecord import read_and_decode
import Variables
import tensorflow as tf

if __name__ == '__main__':
    # with tf.Session() as sess:
    #     sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     try:
    #         step=0
    #         while not coord.should_stop():
    #             # Run training steps or whatever
    #             loss_value=sess.run(train())
    #             step+=1
    #
    #     except tf.errors.OutOfRangeError:
    #         print('Done training -- epoch limit reached')
    #     finally:
    #         # When done, ask the threads to stop.
    #         coord.request_stop()
    #
    #     # Wait for threads to finish.
    #     coord.join(threads)
    #images, labels = read_and_decode("TRAIN.tf")
    images, labels = read_and_decode("TRAIN.tf")
    x_int, ys_int = tf.train.shuffle_batch([images, labels],
                                           batch_size=Variables.BATCH_SIZE,
                                           num_threads=4,
                                           capacity=55000,
                                           min_after_dequeue=50000)

    x=tf.to_float(x_int)
    y_ = tf.one_hot(ys_int, Variables.NUM_CLASSES)
    train(x=x,y_=y_,drop_out=True)