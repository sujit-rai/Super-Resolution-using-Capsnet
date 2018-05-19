import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
from scipy.misc import imsave
import scipy.io
#reading input data
# mnist = input_data.read_data_sets("/home/sujit/capsnet/data/")
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
print(inpu)

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
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
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

#final output after 2 iterations of prediction
caps2_output = caps2_output_round_2


decoder_input = tf.reshape(caps2_output,
                           [-1, 10, 7, 7],
                           name="decoder_input")

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


n_iterations = 100000
restore_checkpoint = True


checkpoint_path = "/home/sujit/capsnet/checkpoint/modeldeconv_pipelined_corrected3.ckpt"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for iteration in range(1, n_iterations):
            
        # Run the training operation and measure the loss:
        _, loss_train, reconstructions, org = sess.run([training_op, reconstruction_loss, decoder_output, X])


        if(iteration%30 == 0):
            print("\rIteration: {}  Loss: {:.5f}".format(
                    iteration,
                    loss_train),
                end="")

            scipy.io.savemat('W.mat', mdict={'org': org,'reconstructions': reconstructions})
            # imsave("recon.png",reconstructions[0,:,:,:])
            save_path = saver.save(sess, checkpoint_path)


    coord.request_stop()
    coord.join(threads)
        # At the end of each epoch,
        # measure the validation loss and accuracy:
        # loss_vals = []
        # acc_vals = []
        # for iteration in range(1, n_iterations_validation + 1):
        #     X_batch, y_batch = mnist.validation.next_batch(batch_size)
        #     loss_val, acc_val = sess.run(
        #             [loss, accuracy],
        #             feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
        #                        y: y_batch})
        #     loss_vals.append(loss_val)
        #     acc_vals.append(acc_val)
        #     print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
        #               iteration, n_iterations_validation,
        #               iteration * 100 / n_iterations_validation),
        #           end=" " * 10)
        # loss_val = np.mean(loss_vals)
        # acc_val = np.mean(acc_vals)
        # print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
        #     epoch + 1, acc_val * 100, loss_val,
        #     " (improved)" if loss_val < best_loss_val else ""))

        # # And save the model if it improved:
        # if loss_val < best_loss_val:
        #     save_path = saver.save(sess, checkpoint_path)
        #     best_loss_val = loss_val


