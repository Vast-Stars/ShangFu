from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import input_Queue
import tf_train

TRAIN_IMAGES = ["./dataset/dataset_1/" + i for i in os.listdir('./dataset/dataset_1')] \
               + ["./dataset/dataset_2/" + i for i in os.listdir('./dataset/dataset_2/')]
# 种类
NUM_CLASSES = 2
BATCH_SIZE = 10000
IMAGE_SIZE = 128
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

TRAIN_LABELS = './dataset/train_labels.txt'
TEST_IMAGES = './dataset/dataset_test/'
TEST_LABELS = './dataset/test_labels.txt'


if __name__ == '__main__':
    #input_Queue.input_pipeline(TRAIN_IMAGES,BATCH_SIZE,read_threads=10)
    img, label = read_and_decode("train.tfrecords")
    # 使用shuffle_batch可以随机打乱输入
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for
