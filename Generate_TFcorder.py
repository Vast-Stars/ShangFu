import tensorflow as tf
import numpy
from PIL import Image

def generate_TFrecord(files_list, labels_list, to_file="train.tfrecords"):
    """
    生成一个TFrecord文件
    :param files_list: 文件列表，例如 ['1.jpg','/dataset/2.jpg']
    :param labels_list: 和文件列表对应的标签列表,例如[0,4]
    :param to_file: 写入目标TFrecord文件
    :return: NULL
    """
    writer = tf.python_io.TFRecordWriter(to_file)
    for (file, label) in zip(files_list, labels_list):
        img = Image.open(file)
        img_raw = img.tobytes()  # 转成byte
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())  # 写入TFrecord文件
    writer.close()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# if __name__ == '__main__':
#     list1 = [
#         './dataset/dataset_1/car_0906.jpg',
#         './dataset/dataset_1/car_0907.jpg',
#         './dataset/dataset_1/car_0908.jpg',
#         './dataset/dataset_1/car_0909.jpg',
#         './dataset/dataset_1/car_0910.jpg',
#         './dataset/dataset_1/car_0911.jpg',
#         './dataset/dataset_1/car_0912.jpg',
#         './dataset/dataset_1/car_0913.jpg',
#         './dataset/dataset_1/car_0914.jpg',
#         './dataset/dataset_1/car_0915.jpg',
#         './dataset/dataset_1/car_0916.jpg',
#         './dataset/dataset_1/car_0917.jpg',
#         './dataset/dataset_1/car_0918.jpg',
#         './dataset/dataset_1/car_0919.jpg',
#         './dataset/dataset_1/car_0920.jpg',
#         './dataset/dataset_1/car_0921.jpg',
#         './dataset/dataset_1/car_0922.jpg',
#         './dataset/dataset_1/car_0923.jpg',
#         './dataset/dataset_1/car_0924.jpg',
#         './dataset/dataset_1/car_0925.jpg',
#         './dataset/dataset_1/car_0926.jpg',
#         './dataset/dataset_1/car_0927.jpg',
#         './dataset/dataset_1/car_0928.jpg',
#         './dataset/dataset_1/car_0929.jpg',
#         './dataset/dataset_1/car_0930.jpg',
#         './dataset/dataset_1/car_0931.jpg',
#         './dataset/dataset_1/car_0932.jpg',
#         './dataset/dataset_1/car_0933.jpg',
#         './dataset/dataset_1/car_0934.jpg',
#         './dataset/dataset_1/car_0935.jpg']
#     list2 = [i for i in range(30)]
#     TEST_generate_TFrecord(list1,list2,'test3.tf')
#     generate_TFrecord(list1,list2,'generate_TF1.tf')
#     TEST2(list1,[list2],'test2.tf')
