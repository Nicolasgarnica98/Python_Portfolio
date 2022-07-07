import skimage.io as io
import os
import skimage.io as io
import matplotlib.pyplot as plt 
from skimage.measure import label
import numpy as np
import scipy.ndimage as ndi
import glob
import re
import cv2
from skimage.color import rgb2gray
from skimage.morphology import dilation, erosion, h_minima
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu

img = []
for i in range(0,400):
    img.append(np.random.randint(100))

img = np.reshape(img,(20,20))

# plt.imshow(img,'gray')
# plt.show()

minimos = h_minima(img,40)
plt.imshow(minimos,'gray')
plt.show()