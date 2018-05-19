import cv2
import numpy as np
from scipy.misc import imresize
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.misc import toimage

# Load an color image in grayscale
img = cv2.imread('test.png')



image = np.array(img)

img = Image.fromarray(image)

img.show()

n = 16

i_width, i_height, i_depth = image.shape[0], image.shape[1], image.shape[2]

print(image.shape)

stride = 8

output_w = n*2
output_h = n*2

output = np.zeros([i_width*2,i_height*2,3],dtype='int')

for z in range(0,3):
    for y in range(0,i_height-n,stride):
        for x in range(0,i_width-n,stride):
            temp = image[x:x+n,y:y+n,z]
            res = imresize(temp,[n*2,n*2],interp='bilinear',mode=None)
            # print(res)
            if x==0 and y==0:
                output[x:output_w,y:output_h,z] = res
                continue
            overlapindex = output[x*2:(x*2)+output_w,y*2:(y*2)+output_h,z] > 0



            overlap = output[x*2:(x*2)+output_w,y*2:(y*2)+output_h,z] + res
            overlap[overlapindex] = overlap[overlapindex]/2
            output[x*2:(x*2)+output_w,y*2:(y*2)+output_h,z] = overlap

print(output)

toimage(output).show()
# imsave(str(index)+".png",output)