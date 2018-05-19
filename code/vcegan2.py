import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
from scipy.misc import imsave
import scipy.io
from scipy.misc import toimage
#reading input data
data_path = os.getcwd() + "/data"
save_path = os.getcwd() + "/checkpoint/vcegan2.ckpt"
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
conv1 = tf.layers.conv2d(X1, name="enc_conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="enc_conv2", **conv2_params)

caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="enc_caps1_raw")

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


caps1_output = squash(caps1_raw, name="enc_caps1_output")

caps2_n_caps = 10
caps2_n_dims = 25

init_sigma = 0.1

# part to whole relationship mapping using W_init

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="enc_W_init")
W = tf.Variable(W_init, name="enc_W")

batch_size = tf.shape(X)[0]

#final weight
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="enc_W_tiled")

caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")

#final primary caps output
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="enc_caps1_output_tiled")


#final predictions for second caps
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="enc_caps2_predicted")


#bij matrix
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="enc_raw_weights")

# Round 1

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="enc_routing_weights")

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




flat_caps2 = tf.layers.flatten(caps2_output_round_2)

fc1 = tf.nn.relu(tf.layers.dense(flat_caps2, 250, name="enc_fc1"))



# print(fc1)
# print(tf.squeeze(caps2_output_round_2))



def generator(inpu, reuse=False):
	with tf.variable_scope("generate") as scope:

		if reuse:
			scope.reuse_variables()

		fulc = tf.nn.relu(tf.layers.dense(inpu, 490))
		reshp = tf.reshape(fulc,[-1,7,7,10])
		hidden1 = tf.nn.relu(tf.layers.conv2d_transpose(reshp, 128, [5,5], strides=(1,1), padding="VALID",name="hidden1"))
		hidden2 = tf.nn.relu(tf.layers.conv2d_transpose(hidden1, 64, [5,5], strides=(1,1),padding="VALID",name="hidden2"))
		hidden3 = tf.nn.relu(tf.layers.conv2d_transpose(hidden2, 32, [5,5], strides=(1,1),padding="VALID",name="hidden3"))
		hidden4 = tf.nn.relu(tf.layers.conv2d_transpose(hidden3, 16, [5,5], strides=(1,1),padding="VALID",name="hidden4"))
		hidden5 = tf.nn.relu(tf.layers.conv2d_transpose(hidden4, 8, [4,4], strides=(1,1),padding="VALID",name="hidden5"))
		gen_output = tf.layers.conv2d_transpose(hidden5,3,[3,3], strides=(1,1), padding="VALID",name="gen_output")

		return tf.nn.tanh(gen_output)


def discriminator(inpu):
	with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:

		dconv1 = tf.nn.relu(tf.layers.conv2d(inpu,name="dconv1",filters=64,kernel_size=3,strides=2,padding="VALID"))
		dconv2 = tf.nn.relu(tf.layers.conv2d(dconv1,name="dconv2",filters=256,kernel_size=3,strides=1,padding="VALID"))
		dconv3 = tf.nn.relu(tf.layers.conv2d(dconv2,name="dconv3",filters=512,kernel_size=3,strides=1,padding="VALID"))
		dconv4 = tf.layers.conv2d(dconv3,name="dconv4",filters=512,kernel_size=3,strides=1,padding="VALID")
		
		dconv5 = tf.nn.relu(tf.layers.conv2d(dconv4,name="dconv5",filters=256,kernel_size=3,strides=1,padding="VALID"))
		dconv6 = tf.layers.flatten(tf.nn.relu(tf.layers.conv2d(dconv5,name="dconv6",filters=128,kernel_size=3,strides=1,padding="VALID")))

		dfc = tf.nn.relu(tf.layers.dense(dconv6,256))

		disc_output = tf.nn.sigmoid(tf.layers.dense(dfc,1))


		return disc_output



def NLLNormal(pred, target):
	c = -0.5 * tf.log(2 * np.pi)
	multiplier = 1.0 / (2.0 * 1)
	tmp = tf.square(pred - target)
	tmp *= -multiplier
	tmp += c
	return tmp

x_tilde = generator(fc1, reuse=False)


real = discriminator(X)

print(real)

fake = discriminator(x_tilde)

print(fake)



# kl_loss = -0.5 * tf.reduce_sum(1 + z_sigma - tf.pow(z_mean, 2) - tf.exp(z_sigma))






D_loss = -tf.reduce_mean(tf.log(real + 1e-12) + tf.log((1. - fake) + 1e-12))


generation_loss = tf.reduce_mean(tf.square(X - x_tilde))

# latent_loss = - 0.5 * tf.reduce_sum(1 + z_sigma - tf.square(z_mean) - tf.square(tf.exp(z_sigma)))

G_loss = -tf.reduce_mean(tf.log(fake + 1e-12))


total_loss = generation_loss



t_variables = tf.trainable_variables()

# encoder_variables = [var for var in t_variables if 'enc_' in var.name]
generator_variables = [var for var in t_variables if 'generate' or 'enc_' in var.name]
discriminator_variables = [var for var in t_variables if 'discri' in var.name]



# global_step = tf.Variable(0, trainable=False)
# add_global = global_step.assign_add(1)
# new_learning_rate = tf.train.exponential_decay(0.0003, global_step=global_step, decay_steps=10000,
#                                                    decay_rate=0.98)

d_solver = tf.train.GradientDescentOptimizer(learning_rate = 1e-4).minimize(D_loss,var_list=discriminator_variables)

e_solver = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(total_loss,var_list=generator_variables)

g_solver = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(G_loss,var_list=generator_variables)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

restore_check = False

with tf.Session() as sess:
	
	if restore_check and tf.train.checkpoint_exists(save_path):
		saver.restore(sess,save_path)
	else:
		sess.run(init)



	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)


	for iterations in range(10000*3000):

		if(iterations<10000):
			_,tlloss,recon, I = sess.run([e_solver, total_loss, x_tilde, X])
			if(iterations%30 == 0):
				print("patches: {}  tloss: {:.5f}".format(iterations, tlloss))
				imsave(os.getcwd()+"/recon.jpg",recon[0])
				imsave(os.getcwd()+"/og.jpg",I[0])
				continue



		_,tlloss = sess.run([e_solver, total_loss])
		_,d_loss = sess.run([d_solver, D_loss])
		_, g_loss, recon, I = sess.run([g_solver, G_loss, x_tilde, X])

		# new_learn_rate = sess.run(new_learning_rate)

		# if new_learn_rate > 0.00005:
		# 	sess.run(add_global)


		if(iterations%30 == 0):
			print("\rPatches: {}  gLoss: {:.5f} dLoss: {:.5f}".format(
                    iterations,
                    g_loss,
                    d_loss))
			save = saver.save(sess,save_path)
			imsave(os.getcwd()+"/recon.jpg",recon[0])
			imsave(os.getcwd()+"/og.jpg",I[0])

	coord.request_stop()
	coord.join(threads)