import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2

#reading input data
# mnist = input_data.read_data_sets("/home/sujit/capsnet/data/")


#input images
X = tf.placeholder(shape=[None, 28, 28, 3], dtype=tf.float32, name="X")


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
    decoder_output = tf.layers.conv2d_transpose(hidden5, 1, [3,3], strides = (1,1), padding = "VALID",
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")


squared_difference = tf.square(X - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")




optimizer = tf.train.AdamOptimizer()

training_op = optimizer.minimize(reconstruction_loss, name="training_op")

init = tf.global_variables_initializer()

saver = tf.train.Saver()


restore_checkpoint = True

checkpoint_path = "/home/sujit/capsnet/checkpoint/modeldeconv_naive.ckpt"

PATH = "/home/sujit/Downloads/DIV2K_train_HR"

files = list()

for path, dirs, filess in os.walk(PATH):
    files = filess


index = 0


def readinput():
    global index
    fullpath = os.path.join(PATH, files[index%len(files)])
    index = index + 1
    img = cv2.imread(fullpath)

    image = np.array(img)

    n = 28

    strd = 28
    finput = []
    np.array(finput)
    instance = np.zeros([1,28,28,3])
    ind = 0
    for y in range(0,image.shape[1]-n,strd):
        for x in range(0,image.shape[0]-n,strd):
            if ind==0:
                # print("inside if")
                instance[0,:,:,:] = image[x:x+n,y:y+n,:]
                ind = ind + 1
                finput = instance
            else:
                instance[0,:,:,:] = image[x:x+n,y:y+n,:]
                finput = np.concatenate((instance,finput),axis=0)

    return finput

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for iteration in range(100000):
        X_batch = readinput()
        # Run the training operation and measure the loss:
        _, loss_train = sess.run(
            [training_op, reconstruction_loss],
            feed_dict={X: X_batch.reshape([-1, 28, 28, 3])})

        if(iteration%30 == 0):
          print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                  iteration, n_iterations_per_epoch,
                  iteration * 100 / n_iterations_per_epoch,
                  loss_train),
              end="")
    
          save_path = saver.save(sess, checkpoint_path)

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


