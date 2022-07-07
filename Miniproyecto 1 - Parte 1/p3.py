from skimage.filters import threshold_otsu
from  scipy.io import loadmat
import os
import glob
import nibabel 
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

image = data.camera()
thresh = threshold_otsu(image)
print(thresh)

num = 0
for i in range(0,len(image.flatten())):
    if image.flatten()[i] == thresh:
        num = i
        break
print(num)

binary = image >= thresh
print(int(binary.flatten()[num]))




fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].hist(image.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')

plt.show()