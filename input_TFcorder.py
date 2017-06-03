from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy
import os
from PIL import Image

# --------------------------------------------------------------
TRAIN_IMAGES = ["./dataset/dataset_1/" + i for i in os.listdir('./dataset/dataset_1')] \
               + ["./dataset/dataset_2/" + i for i in os.listdir('./dataset/dataset_2/')]
TRAIN_LABELS = './dataset/train_labels.txt'
TEST_IMAGES = './dataset/dataset_test/'
TEST_LABELS = './dataset/test_labels.txt'
# --------------------------------------------------------------


IMAGE_SIZE = 128
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

writer = tf.python_io.TFRecordWriter("train.tfrecords")
for file in TRAIN_IMAGES:
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



