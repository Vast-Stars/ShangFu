import tensorflow as tf
import numpy
from PIL import Image


def generate_TFrecord(files_list,labels_list,to_file="train.tfrecords"):
    """
    生成一个TFrecord文件
    :param files_list: 文件列表。
    :param labels_list: 和文件列表对应的标签列表。
    :param to_file: 写入目标TFrecord文件
    :return: NULL
    """
    writer = tf.python_io.TFRecordWriter(to_file)
    for (file,label) in zip(files_list,labels_list):
        img = Image.open(file)
        img_raw = img.tobytes()  #转成byte
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())#写入TFrecord文件
    writer.close()

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



