import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
from scipy.misc import imsave
import scipy.io
#reading input data
data_path = os.getcwd() + "/data"
filename_list = []
for path,dirs,files in os.walk(data_path):
  for file in files:
    filename_list.append(os.path.join(data_path,file))

filename_queue = tf.train.string_input_producer(filename_list)

image_reader = tf.WholeFileReader()

_, image_file = image_reader.read(filename_queue)

img = tf.expand_dims(tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(image_file),28,28),axis=0)

# print(img)

X = tf.cast(tf.train.shuffle_batch([img], 65, 100, 90, enqueue_many=True, shapes=[28,28,3],num_threads=1),tf.float32)

X1 = tf.image.resize_images(X,[14,14])

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
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 1,
    "padding": "same",
    "activation": tf.nn.relu
}


#primary caps layer
conv1 = tf.layers.conv2d(X1, name="conv1", **conv1_params)
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
caps2_n_dims = 25

init_sigma = 0.1

# part to whole relationship mapping using W_init

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

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










e_input = tf.reshape(tf.squeeze(caps2_output_round_2),[-1,250],name="e_input")


n_hidden1 = 786

n_hidden2 = 1024

n_hidden3 = 786

n_hidden4 = 490

e_hidden1 = tf.layers.dense(e_input, n_hidden1, activation=tf.nn.relu, name="e_hidden1")

e_hidden2 = tf.layers.dense(e_hidden1, n_hidden2, activation=tf.nn.relu, name="e_hidden2")

e_hidden3 = tf.layers.dense(e_hidden2, n_hidden3, activation=tf.nn.relu, name="e_hidden3")

e_output = tf.layers.dense(e_hidden3, n_hidden4, activation=tf.nn.relu, name="e_hidden4")

# hr_output = tf.reshape(tf.squeeze(caps2_output_round_2),[-1,490],name="hr_output")

decoder_input = tf.reshape(e_output,[-1,10,7,7],name="decoder_input")

decoder_input = tf.transpose(decoder_input, perm=[0,2,3,1])

with tf.name_scope("decoder"):
    hidden1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(decoder_input, 128, [5,5], strides = (1,1), padding = "VALID",
                              activation=tf.nn.relu,
                              name="hidden1"))
    hidden2 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(hidden1, 64, [5,5], strides = (1,1), padding = "VALID",
                              activation=tf.nn.relu,
                              name="hidden2"))
    hidden3 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(hidden2, 32, [5,5], strides = (1,1), padding = "VALID",
                              activation=tf.nn.relu,
                              name="hidden3"))
    hidden4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(hidden3, 16, [5,5], strides = (1,1), padding = "VALID",
                              activation=tf.nn.relu,
                              name="hidden4"))
    hidden5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(hidden4, 8, [4,4], strides = (1,1), padding = "VALID",
                              activation=tf.nn.relu,
                              name="hidden5"))
    decoder_output = tf.layers.conv2d_transpose(hidden5, 3, [3,3], strides = (1,1), padding = "VALID",
                                     activation=tf.nn.relu,
                                     name="decoder_output")


squared_difference = tf.square(X - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")




optimizer = tf.train.AdamOptimizer()

training_op = optimizer.minimize(reconstruction_loss, name="training_op")

init = tf.global_variables_initializer()

saver = tf.train.Saver()


n_iterations = 3000 * 10000
restore_checkpoint = True


checkpoint_path1 = os.getcwd() + "/checkpoint/modeldeconv_pipelined_corrected_lr1.ckpt"

checkpoint_path2 = os.getcwd() + "/checkpoint/modeldeconv_combined.ckpt"

checkpoint_path3 = os.getcwd() + "/checkpoint/modeldeconv_pipelined_corrected3.ckpt"

checkpoint_path = os.getcwd() + "/checkpoint/modeldeconv_finetune.ckpt"

checkpoint_pathe = os.getcwd() + "/checkpoint/modeldeconv_e.ckpt"

checkpoint_path_recent = os.getcwd() + "/checkpoint/model_recent.ckpt"

init = tf.global_variables_initializer()

reuse_vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="conv[1]") 
reuse_vars_dict1 = dict([(var.op.name, var) for var in reuse_vars1])

reuse_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="conv[2]")

reuse_vars_dict2 = dict([(var.op.name, var) for var in reuse_vars2])

reuse_vars_dict3 = dict([("W",W)])

reuse_vars_dict1.update(reuse_vars_dict2)

reuse_vars_dict1.update(reuse_vars_dict3)

saver1 = tf.train.Saver(reuse_vars_dict1)

reuse_vars_e1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="e_hidden1")

reuse_vars_edict1 = dict([(var.op.name, var) for var in reuse_vars_e1])

reuse_vars_e2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="e_hidden2")

reuse_vars_edict2 = dict([(var.op.name, var) for var in reuse_vars_e2])

reuse_vars_e3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="e_hidden3")

reuse_vars_edict3 = dict([(var.op.name, var) for var in reuse_vars_e3])

reuse_vars_e4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="e_hidden4")

reuse_vars_edict4 = dict([(var.op.name, var) for var in reuse_vars_e4])

reuse_vars_edict1.update(reuse_vars_edict2)

reuse_vars_edict1.update(reuse_vars_edict3)

reuse_vars_edict1.update(reuse_vars_edict4)

saver2 = tf.train.Saver(reuse_vars_edict1)

reuse_vars_d1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden1")

reuse_vars_ddict1 = dict([(var.op.name, var) for var in reuse_vars_d1])

reuse_vars_d2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden2")

reuse_vars_ddict2 = dict([(var.op.name, var) for var in reuse_vars_d2])

reuse_vars_d3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden3")

reuse_vars_ddict3 = dict([(var.op.name, var) for var in reuse_vars_d3])

reuse_vars_d4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden4")

reuse_vars_ddict4 = dict([(var.op.name, var) for var in reuse_vars_d4])

reuse_vars_d5 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden5")

reuse_vars_ddict5 = dict([(var.op.name, var) for var in reuse_vars_d5])

reuse_vars_d6 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="decoder_output")

reuse_vars_ddict6 = dict([(var.op.name, var) for var in reuse_vars_d6])

reuse_vars_ddict1.update(reuse_vars_ddict2)

reuse_vars_ddict1.update(reuse_vars_ddict3)

reuse_vars_ddict1.update(reuse_vars_ddict4)

reuse_vars_ddict1.update(reuse_vars_ddict5)

reuse_vars_ddict1.update(reuse_vars_ddict6)

saver3 = tf.train.Saver(reuse_vars_ddict1)



saver = tf.train.Saver()



with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        # saver.restore(sess, checkpoint_path)
        sess.run(init)
        saver1.restore(sess, checkpoint_path1)
        saver2.restore(sess, checkpoint_path2)
        saver3.restore(sess, checkpoint_path3)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for iteration in range(1, n_iterations):
            
        # Run the training operation and measure the loss:
        _, loss_train, reconstructions, org = sess.run([training_op, reconstruction_loss, decoder_output, X1])


        if(iteration%30 == 0):
            print "\rPatches: {}  Loss: {:.5f}".format(
                    iteration,
                    loss_train)

            scipy.io.savemat('W.mat', mdict={'org': org,'reconstructions': reconstructions})
            # imsave("recon.png",reconstructions[0,:,:,:])
            save_path = saver.save(sess, checkpoint_path_recent)


    coord.request_stop()
    coord.join(threads)