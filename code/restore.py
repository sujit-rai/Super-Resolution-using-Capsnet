import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
v4 = tf.get_variable("v4", shape=[1,3])
# v5 = tf.get_variable("v5", shape=[5])

# v3 = tf.get_variable("v3", shape=[5], initializer = tf.zeros_initializer)

# dec_v3 = v3.assign(v3-3)


# # Add ops to save and restore all the variables.
# saver = tf.train.Saver({"v1": v4,"v2": v5})

# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   v3.initializer.run()
#   sess.run(tf.global_variables_initializer())
#   # saver.restore(sess, "/tmp/model.ckpt")
#   print("Model restored.")
#   # Check the values of the variables
#   dec_v3.op.run()
#   print("v1 : %s" % v4.eval())
#   print("v2 : %s" % v5.eval())
#   print("v3 : %s" % v3.eval())


with tf.name_scope("test"):
	hidden1 = tf.layers.dense(v4,32,name="hidden1")


variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="hidden1")

variables_dict = dict([(var.op.name, var) for var in variables])

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	print(sess.run(variables_dict))



