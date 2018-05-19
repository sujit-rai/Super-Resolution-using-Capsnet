import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
from scipy.misc import imsave
import scipy.io
#reading input data
# mnist = input_data.read_data_sets("/home/sujit/capsnet/data/")
data_path = "/home/sujit/Downloads/lr_test"
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

inpu = image_to_patches(my_img,14,14)

X = inpu

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(X))

