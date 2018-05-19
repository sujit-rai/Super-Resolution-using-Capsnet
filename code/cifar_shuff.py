import tensorflow as tf
import os
data_path = "/home/sujit/Downloads/DIV2K_train_HR"







def cifar_shuffle_batch():
    num_threads = 1
    batch_size = 65
    # create a list of all our filenames
    filename_list = []

    for path,dirs,files in os.walk(data_path):
        for file in files:
            filename_list.append(os.path.join(data_path,file))

    # create a filename queue
    file_q = cifar_filename_queue(filename_list)
    # file_q = tf.train.string_input_producer(filename_list)
    # read the data - this contains a FixedLengthRecordReader object which handles the
    # de-queueing of the files.  It returns a processed image and label, with shapes
    # ready for a convolutional neural network
    image = read_data(file_q)

    print(image)
    # setup minimum number of examples that can remain in the queue after dequeuing before blocking
    # occurs (i.e. enqueuing is forced) - the higher the number the better the mixing but
    # longer initial load time
    min_after_dequeue = 100
    # setup the capacity of the queue - this is based on recommendations by TensorFlow to ensure
    # good mixing
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    # image_batch, label_batch = cifar_shuffle_queue_batch(image, label, batch_size, num_threads)
    image_batch = tf.train.shuffle_batch([image], batch_size, capacity, min_after_dequeue,
                                                      num_threads=num_threads)
    # now run the training
    cifar_run(image_batch)


def cifar_run(image):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            image_batch = sess.run([image])
            print(image_batch.shape)

        coord.request_stop()
        coord.join(threads)


def cifar_filename_queue(filename_list):
    # convert the list to a tensor
    string_tensor = tf.convert_to_tensor(filename_list, dtype=tf.string)
    # randomize the tensor
    tf.random_shuffle(string_tensor)
    # create the queue
    fq = tf.FIFOQueue(capacity=10, dtypes=tf.string)
    # create our enqueue_op for this q
    fq_enqueue_op = fq.enqueue_many([string_tensor])
    # create a QueueRunner and add to queue runner list
    # we only need one thread for this simple queue
    tf.train.add_queue_runner(tf.train.QueueRunner(fq, [fq_enqueue_op] * 1))
    return fq


def cifar_shuffle_queue_batch(image, label, batch_size, capacity, min_after_dequeue, threads):
    tensor_list = [image, label]
    dtypes = [tf.float32, tf.int32]
    shapes = [image.get_shape(), label.get_shape()]
    q = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_after_dequeue,
                              dtypes=dtypes, shapes=shapes)
    enqueue_op = q.enqueue(tensor_list)
    # add to the queue runner
    tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op] * threads))
    # now extract the batch
    image_batch, label_batch = q.dequeue_many(batch_size)
    return image_batch, label_batch

def read_data(file_q):
    # Code from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_q)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    # result.label = tf.cast(
    #     tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # # The remaining bytes after the label represent the image, which we reshape
    # # from [depth * height * width] to [depth, height, width].
    # depth_major = tf.reshape(
    #     tf.strided_slice(record_bytes, [label_bytes],
    #                      [label_bytes + image_bytes]),
    #     [result.depth, result.height, result.width])
    # # Convert from [depth, height, width] to [height, width, depth].
    # result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    # reshaped_image = tf.cast(result.uint8image, tf.float32)

    # height = 24
    # width = 24

    # # Image processing for evaluation.
    # # Crop the central [height, width] of the image.
    # resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
    #                                                        height, width)

    # # Subtract off the mean and divide by the variance of the pixels.
    # float_image = tf.image.per_image_standardization(resized_image)

    # # Set the shapes of tensors.
    # float_image.set_shape([height, width, 3])
    # result.label.set_shape([1])

    return record_bytes

if __name__ == "__main__":
    run_opt = 3
    if run_opt == 1:
        FIFO_queue_demo_no_coord()
    elif run_opt == 2:
        FIFO_queue_demo_with_coord()
    elif run_opt == 3:
        cifar_shuffle_batch()