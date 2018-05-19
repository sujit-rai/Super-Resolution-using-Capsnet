import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2
from scipy.misc import imsave
import scipy.io
from scipy.misc import toimage
#reading input data
# mnist = input_data.read_data_sets("/home/sujit/capsnet/data/")
PATH = "/home/sujit/capsnet/testimg"

files = list()

for path, dirs, filess in os.walk(PATH):
    files = filess

imqueue = []
imqueue = np.array(imqueue)
index = 0
rdel = np.arange(65)



def readinput():
    global index
    global PATH
    global files
    fullpath = os.path.join(PATH, files[index%len(files)])
    index = index + 1
    img = cv2.imread(fullpath)

    image = np.array(img)

    n = 28

    strd = 28
    finput = []
    finput = np.array(finput)
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



def nextbatch():
    global imqueue
    global rdel
    if(imqueue.shape[0]<1):
        imqueue = readinput()

    if(imqueue.shape[0]<65):
        imqueue = np.concatenate((imqueue,readinput()),axis=0)
    foutput = imqueue[0:64,:,:,:]
    # toimage(imqueue[0]).show()
    imqueue = np.delete(imqueue,np.arange(65), axis=0)
    # toimage(imqueue[0]).show()
    return foutput




#input images
x_input_data = tf.convert_to_tensor(nextbatch(),dtype=tf.float32)


q = tf.FIFOQueue(capacity=1, dtypes=tf.float32) # enqueue 5 batches
# We use the "enqueue" operation so 1 element of the queue is the full batch
enqueue_op = q.enqueue(x_input_data)
numberOfThreads = 1
qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
tf.train.add_queue_runner(qr)
X = q.dequeue() # It replaces our input placeholder



with tf.Session() as sess:
	# coord = tf.train.Coordinator()
	# threads = tf.train.start_queue_runners(coord=coord)
	outputarr = list()
	for i in range(1000):
		output = sess.run(x_input_data)
		outputarr.append(output)
	scipy.io.savemat('W.mat', mdict={'outputarr':outputarr})