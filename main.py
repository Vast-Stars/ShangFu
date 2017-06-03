from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tf_train import train
from read_TFrecord import read_and_decode
# 种类
NUM_CLASSES = 2
BATCH_SIZE = 10000

IMAGE_SIZE = 128
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
TRAIN_EPOCHS=10000
SAVE_STEP=1000
SHOW_STEP=100

if __name__ == '__main__':
    img, label = read_and_decode("train.tfrecords")
    # 使用shuffle_batch可以随机打乱输入
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=10000, capacity=2000,
                                                    min_after_dequeue=20000)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step=0
            while not coord.should_stop():
                # Run training steps or whatever
                loss_value=sess.run(train())
                step+=1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)