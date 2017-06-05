from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tf_train import train
from read_TFrecord import read_and_decode
import Variables

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

    train(imgages,labels)