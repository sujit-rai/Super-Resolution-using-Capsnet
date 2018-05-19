import cv2
import numpy as np
from scipy.misc import imresize
from PIL import Image
import cv2
from scipy.misc import toimage
import scipy.io

mat = scipy.io.loadmat("res.mat")

res = mat["output"]

i_width = mat["width"][0][0]

i_height = mat["height"][0][0]

og = mat["og"]




print(i_width)

output = np.zeros([i_width,i_height,3])

org = np.zeros([int(i_width/2),int(i_height/2),3])

n = 28

stride = 28

o_width = i_width

o_height = i_height

print(res.shape)

index = 0
for x in range(0,o_width-n,stride):
	for y in range(0,o_height-n,stride):
		org[x:x+14,y:y+14,:] = og[index,0,:,:,:]
		output[x:x+n,y:y+n,:] = res[index,0,:,:,:]
		index = index + 1


toimage(output).show()

toimage(org).show()