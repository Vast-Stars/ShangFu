import tensorflow as tf
import numpy
from PIL import Image
import Variables

def generate_TFrecord(files_list, labels_list, to_file="train.tfrecords",is_one_hot=False):
    """
    生成一个TFrecord文件
    :param files_list: 文件列表，例如 ['1.jpg','/dataset/2.jpg']
    :param labels_list: 和文件列表对应的标签列表,例如[0,4]
    :param to_file: 写入目标TFrecord文件
    :return: NULL
    """
    # tem=numpy.array(labels_list, dtype=int)
    # labels_list=dense_to_one_hot(numpy.array(labels_list,dtype=int) ,Variables.NUM_CLASSES)

    writer = tf.python_io.TFRecordWriter(to_file)
    i=0
    for (file, label) in zip(files_list, labels_list):
        print(i)
        i+=1
        img = Image.open(file)
        img_raw = img.tobytes()  # 转成byte
        #tem=label.tolist()
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
    labels_one_hot = numpy.zeros((num_labels, num_classes),dtype=int)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


if __name__ == '__main__':
    import os
    a=os.listdir('DATA/CARS/')
    list1= ['DATA/CARS/'+ i for i in a]
    a=os.listdir('DATA/HUMAN/')
    list2= ['DATA/HUMAN/'+ i for i in a]
    #     list2 = [i for i in range(30)]
    list_label = [0 ]*len(list1)+[1]*len(list2)
    generate_TFrecord(list1+list2, list_label, 'TRAIN.tf')  # TEST2(list1,[list2],'test2.tf')
