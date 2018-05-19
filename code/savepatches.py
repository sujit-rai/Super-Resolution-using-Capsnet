import cv2
import numpy as np
from scipy.misc import imresize
from PIL import Image
import cv2
from scipy.misc import toimage
import scipy.io
import os
from scipy.misc import imread
from scipy.misc import imsave

PATH = "/home/sujit/capsnet/Set14"
savep = "/home/sujit/capsnet/data"
files = []
index = 0
for path, dirs, filess in os.walk(PATH):
	for filename in filess:
		fullpath = os.path.join(path, filename)
		img = imread(fullpath)
		# toimage(img).show()
		# input()
		n = 28

		strd = 14
		# print(img.shape)
		for y in range(0,img.shape[1]-n,strd):
			for x in range(0,img.shape[0]-n,strd):
				index = index + 1
				# toimage(img[x:x+n,y:y+n,:]).show()
				imsave(savep+'/'+str(index)+".jpg",img[x:x+n,y:y+n,:])
				# input()
