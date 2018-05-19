import tensorflow as tf
import numpy as np
import os
import scipy.io

from PIL import Image
data_path = "/home/sujit/capsnet/testimg"
filename_list = []

for path,dirs,files in os.walk(data_path):
  for file in files:
    filename_list.append(os.path.join(data_path,file))

filename_queue = tf.train.string_input_producer(filename_list)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value)





def image_to_patches(image, patch_height, patch_width):
    # resize image so that it's dimensions are dividable by patch_height and patch_width
    image_height = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    image_width = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    height = tf.cast(tf.ceil(image_height / patch_height) * patch_height, dtype=tf.int32)
    width = tf.cast(tf.ceil(image_width / patch_width) * patch_width, dtype=tf.int32)

    num_rows = height // patch_height
    num_cols = width // patch_width
    # make zero-padding
    image = tf.squeeze(tf.image.resize_image_with_crop_or_pad(image, height, width))

    # get slices along the 0-th axis
    image = tf.reshape(image, [num_rows, patch_height, width, -1])
    # h/patch_h, w, patch_h, c
    image = tf.transpose(image, [0, 2, 1, 3])
    # get slices along the 1-st axis
    # h/patch_h, w/patch_w, patch_w,patch_h, c
    image = tf.reshape(image, [num_rows, num_cols, patch_width, patch_height, -1])
    # num_patches, patch_w, patch_h, c
    image = tf.reshape(image, [num_rows * num_cols, patch_width, patch_height, -1])
    # num_patches, patch_h, patch_w, c
    return tf.transpose(image, [0, 2, 1, 3])

X = image_to_patches(my_img,28,28)

# q = tf.FIFOQueue(capacity=10, dtypes=tf.uint8)

# enqueue_op = q.enqueue(X)

# qr = tf.train.QueueRunner(q,[enqueue_op]*1)
# tf.train.add_queue_runner(qr)
# inpu = q.dequeue()


inpu = tf.train.shuffle_batch([X], 65, 100, 90, enqueue_many=True, shapes=[28,28,3],num_threads=1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  outputarr = list()
  for i in range(10000):
    im,m = sess.run([inpu,my_img])
    print(i)
    # outputarr.append(im)

  # scipy.io.savemat('W.mat',mdict={'outputarr':outputarr})

  coord.request_stop()
  coord.join(threads)