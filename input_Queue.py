
import tensorflow as tf
import numpy



def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    key, file = reader.read(filename_queue)
    uint8image = tf.image.decode_jpeg(file, channels=3)

    processed_example =serialized_example
    label = list(map(lambda x: 0 if (x.find('dataset_1') != -1) else 1, serialized_example))
    return processed_example, label


def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [read_my_file_format(filename_queue)
                    for _ in range(read_threads)]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
        example_list,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot




