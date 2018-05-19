# import tensorflow as tf
import numpy as np
import scipy.io
from scipy.misc import toimage
mat = scipy.io.loadmat('W.mat')
org = mat['org']
reconstructions = mat['reconstructions']

print(org.shape)
print(reconstructions.shape)

toimage(org[63,:,:,:]).show()
toimage(reconstructions[63,:,:,:]).show()