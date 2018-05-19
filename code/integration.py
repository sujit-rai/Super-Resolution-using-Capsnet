import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
from scipy.misc import imsave
import scipy.io


data_path = "/home/sujit/Downloads/DIV2K_train_HR"
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

inpu = image_to_patches(my_img,28,28)

# q = tf.FIFOQueue(capacity=10, dtypes=tf.uint8)

# enqueue_op = q.enqueue(X)

# qr = tf.train.QueueRunner(q,[enqueue_op]*1)
# tf.train.add_queue_runner(qr)
# inpu = q.dequeue()


X = tf.cast(tf.train.shuffle_batch([inpu], 65, 100, 90, enqueue_many=True, shapes=[28,28,3],num_threads=1),tf.float32)


#initializing capsule parameters
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8

conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
    "trainable" : False
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu,
    "trainable" : False
}


#primary caps layer
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


caps1_output = squash(caps1_raw, name="caps1_output")

caps2_n_caps = 10
caps2_n_dims = 49

init_sigma = 0.1

# part to whole relationship mapping using W_init

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W", trainable = False)

batch_size = tf.shape(X)[0]

#final weight
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")

#final primary caps output
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")


#final predictions for second caps
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")


#bij matrix
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

# Round 1

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")

caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")

# Round 2

caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")














#low resolution network
X1 = tf.image.resize_images(X,[14,14])


caps11_n_maps = 32
caps11_n_caps = caps11_n_maps * 6 * 6  # 1152 primary capsules
caps11_n_dims = 8

conv11_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
    "trainable" : False
}


conv21_params = {
    "filters": caps11_n_maps * caps11_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 1,
    "padding": "same",
    "activation": tf.nn.relu,
    "trainable" : False
}


#primary caps layer
conv3 = tf.layers.conv2d(X1, name="conv3", **conv11_params)
conv4 = tf.layers.conv2d(conv3, name="conv4", **conv21_params)

caps11_raw = tf.reshape(conv4, [-1, caps11_n_caps, caps11_n_dims],
                       name="caps11_raw")


caps11_output = squash(caps11_raw, name="caps11_output")


caps21_n_caps = 10
caps21_n_dims = 25

W1_init = tf.random_normal(
    shape=(1, caps11_n_caps, caps21_n_caps, caps21_n_dims, caps11_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W1_init")
W1 = tf.Variable(W1_init, name="W1", trainable = False)

batch_size1 = tf.shape(X1)[0]

W1_tiled = tf.tile(W1, [batch_size1, 1, 1, 1, 1], name="W1_tiled")

caps11_output_expanded = tf.expand_dims(caps11_output, -1,
                                       name="caps11_output_expanded")
caps11_output_tile = tf.expand_dims(caps11_output_expanded, 2,
                                   name="caps11_output_tile")

#final primary caps output
caps11_output_tiled = tf.tile(caps11_output_tile, [1, 1, caps21_n_caps, 1, 1],
                             name="caps11_output_tiled")


#final predictions for second caps
caps21_predicted = tf.matmul(W1_tiled, caps11_output_tiled,
                            name="caps21_predicted")


#bij matrix
raw1_weights = tf.zeros([batch_size1, caps11_n_caps, caps21_n_caps, 1, 1],
                       dtype=np.float32, name="raw1_weights")

# Round 1

routing1_weights = tf.nn.softmax(raw1_weights, dim=2, name="routing1_weights")

weighted1_predictions = tf.multiply(routing1_weights, caps21_predicted,
                                   name="weighted1_predictions")
weighted1_sum = tf.reduce_sum(weighted1_predictions, axis=1, keep_dims=True,
                             name="weighted1_sum")

caps21_output_round_1 = squash(weighted1_sum, axis=-2,
                              name="caps21_output_round_1")

# Round 2

caps21_output_round_1_tiled = tf.tile(
    caps21_output_round_1, [1, caps11_n_caps, 1, 1, 1],
    name="caps21_output_round_1_tiled")

agreement1 = tf.matmul(caps21_predicted, caps21_output_round_1_tiled,
                      transpose_a=True, name="agreement1")

raw1_weights_round_2 = tf.add(raw1_weights, agreement1,
                             name="raw1_weights_round_2")

routing1_weights_round_2 = tf.nn.softmax(raw1_weights_round_2,
                                        dim=2,
                                        name="routing1_weights_round_2")
weighted1_predictions_round_2 = tf.multiply(routing1_weights_round_2,
                                           caps21_predicted,
                                           name="weighted1_predictions_round_2")
weighted1_sum_round_2 = tf.reduce_sum(weighted1_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted1_sum_round_2")
caps21_output_round_2 = squash(weighted1_sum_round_2,
                              axis=-2,
                              name="caps21_output_round_2")


print(caps21_output_round_2)





















#low resolution embedding to high resolution embedding



e_input = tf.reshape(tf.squeeze(caps21_output_round_2),[-1,250],name="e_input")


n_hidden1 = 786

n_hidden2 = 1024

n_hidden3 = 786

n_hidden4 = 490

e_hidden1 = tf.layers.dense(e_input, n_hidden1, activation=tf.nn.relu, name="e_hidden1")

e_hidden2 = tf.layers.dense(e_hidden1, n_hidden2, activation=tf.nn.relu, name="e_hidden2")

e_hidden3 = tf.layers.dense(e_hidden2, n_hidden3, activation=tf.nn.relu, name="e_hidden3")

e_output = tf.layers.dense(e_hidden3, n_hidden4, activation=tf.nn.relu, name="e_hidden4")

hr_output = tf.reshape(tf.squeeze(caps2_output_round_2),[-1,490],name="hr_output")

squared_difference = tf.square(e_output - hr_output, name="squared_difference")

loss = tf.reduce_mean(squared_difference,name="loss")

optimizer = tf.train.AdamOptimizer()

training_op = optimizer.minimize(loss, name="training_op")













init = tf.global_variables_initializer()







# eh1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="e_hidden1")

# eh2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="e_hidden2")

# eh3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="e_hidden3")

# eh4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="e_hidden4")



# var_list = [var for var in eh1] + [var for var in eh2] + [var for var in eh3] + [var for var in eh4]

# optimizer_slots = [
#     optimizer.get_slot()
# ]

# var_list = var_list + optimizer_slots

# init = tf.variables_initializer(var_list)

reuse_vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="conv[1]") 
reuse_vars_dict1 = dict([(var.op.name, var) for var in reuse_vars1])

reuse_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="conv[2]")

reuse_vars_dict2 = dict([(var.op.name, var) for var in reuse_vars2])

reuse_vars_dict3 = dict([("W",W)])

reuse_vars_dict1.update(reuse_vars_dict2)

reuse_vars_dict1.update(reuse_vars_dict3)




saver1 = tf.train.Saver(reuse_vars_dict1)

checkpoint_path1 = "/home/sujit/capsnet/checkpoint/modeldeconv_pipelined_corrected3.ckpt"



















reuse_vars11 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv[3]")

reuse_vars_dict11 = dict([(var.op.name.replace("conv3","conv1"),var) for var in reuse_vars11])


reuse_vars21 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="conv[4]")

reuse_vars_dict21 = dict([(var.op.name.replace("conv4","conv2"), var) for var in reuse_vars21])

reuse_vars_dict31 = dict([("W",W1)])

reuse_vars_dict11.update(reuse_vars_dict21)

reuse_vars_dict11.update(reuse_vars_dict31)

print(reuse_vars_dict1)

print(reuse_vars_dict11)

saver2 = tf.train.Saver(reuse_vars_dict11)










saver = tf.train.Saver()


restore_check = False

# variables_names = [v.name for v in tf.trainable_variables()]

n_iterations = 100000

checkpoint_path = "/home/sujit/capsnet/checkpoint/modeldeconv_combined.ckpt"

with tf.Session() as sess:

	sess.run(init)

	saver1.restore(sess, checkpoint_path1)

	saver2.restore(sess,"/home/sujit/capsnet/checkpoint/modeldeconv_pipelined_corrected_lr1.ckpt")

	if restore_check and tf.train.checkpoint_exists(checkpoint_path):
		saver.restore(sess, checkpoint_path)
		

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for iterations in range(1,n_iterations):
		_,tloss = sess.run([training_op,loss])

		if(iterations%100 == 0):
			print("\rIteration: {}  Loss: {:.5f}".format(
                    iterations,
                    tloss),
                end="")
			save_path = saver.save(sess,checkpoint_path)

	coord.request_stop()
	coord.join(threads)
	# values = sess.run(variables_names)
	# for k, v in zip(variables_names, values):
	# 	print("Variable: ", k)
	# 	print("Shape: ", v.shape)
	# 	print(v)


