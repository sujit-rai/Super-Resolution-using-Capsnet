import tensorflow as tf
import numpy as np
c = 3
h = 1024
p = 128


image = tf.random_normal([1,h,h,c])

# Image to Patches Conversion
pad = [[0,0],[0,0]]
patches = tf.space_to_batch_nd(image,[p,p],pad)
patches = tf.split(patches,p*p,0)
patches = tf.stack(patches,3)
patches = tf.reshape(patches,[int((h/p)**2),p,p,c])

# Do processing on patches
# Using patches here to reconstruct
patches_proc = tf.reshape(patches,[1,int(h/p),int(h/p),int(p*p),c])
patches_proc = tf.split(patches_proc,p*p,3)
patches_proc = tf.stack(patches_proc,axis=0)
patches_proc = tf.reshape(patches_proc,[p*p,int(h/p),int(h/p),c])

reconstructed = tf.batch_to_space_nd(patches_proc,[p, p],pad)

sess = tf.Session()
I,P,R_n = sess.run([image,patches,reconstructed])
print(I.shape)
print(P.shape)
print(R_n.shape)
err = np.sum((R_n-I)**2)
print(err)