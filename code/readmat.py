import tensorflow as tf
import numpy as np
import scipy.io
from scipy.misc import toimage
mat = scipy.io.loadmat('W.mat')

print(mat['outputarr'].shape)
# input()

outputarr = mat['outputarr']

for i in range(len(outputarr)):
	curr = outputarr[0][i]
	nxt = outputarr[1][i]
	toimage(curr).show()
	toimage(nxt).show()
	input()