from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy
import os
from PIL import Image

# 种类
NUM_CLASSES = 10

# The images are always 28x28 pixels.
IMAGE_SIZE = 128
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

TRAIN_LABELS = './dataset/train_labels.txt'
TEST_IMAGES = './dataset/dataset_test/'
TEST_LABELS = './dataset/test_labels.txt'
#
# TRAIN_IMAGES=["./dataset/dataset_2/"+i for i in os.listdir('./dataset/dataset_2')] +["./dataset/dataset_3/"+i for i in os.listdir('./dataset/dataset_3/')]

TRAIN_IMAGES1 = ["./dataset/dataset_1/" + i for i in os.listdir('./dataset/dataset_1')]
TRAIN_IMAGES2 = ["./dataset/dataset_2/" + i for i in os.listdir('./dataset/dataset_2')]

# filename_queue1 = tf.train.string_input_producer(TRAIN_IMAGES1, shuffle=True)
# filename_queue2 = tf.train.string_input_producer(TRAIN_IMAGES2, shuffle=True)
writer = tf.python_io.TFRecordWriter("train.tfrecords")
for file in TRAIN_IMAGES1:
    img = Image.open(file)
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    writer.write(example.SerializeToString())
# for file in TRAIN_IMAGES2:
#     img = Image.open(file)
#     img_raw = img.tobytes()
#     example = tf.train.Example(features=tf.train.Features(
#         feature={
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
#             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#         }))
#     writer.write(example.SerializeToString())
writer.close()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_labels(file, one_hot=False, num_classes=NUM_CLASSES):
    with open(file) as f:
        tem = f.read().split()
    tem = [int(x) for x in tem]
    labels = numpy.array(tem)
    if one_hot:
        return dense_to_one_hot(labels, num_classes)
    return labels



