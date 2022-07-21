import numpy as np
import requests
import skimage.io as io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

#Image download
r = requests.get('https://animalesmascotas.com//wp-content/uploads/2016/08/guepardos-familia-de-guepardos.jpg')
with open('Intro_ImageAnalisis/img_cheeta.jpg', 'wb') as f:
      f.write(r.content)

#Image load
img_cheeta = io.imread(os.path.join(('Intro_ImageAnalisis/img_cheeta.jpg')))
print(img_cheeta.shape)

#Red channel
r_channel = img_cheeta.copy()
r_channel[:,:,1] = 0
r_channel[:,:,2] = 0
r_channel = rgb2gray(r_channel)

#Green channel
g_channel = img_cheeta.copy()
g_channel[:,:,0] = 0
g_channel[:,:,2] = 0
g_channel = rgb2gray(g_channel)

#Blue channel
b_channel = img_cheeta.copy()
b_channel[:,:,0] = 0
b_channel[:,:,1] = 0
b_channel = rgb2gray(b_channel)


#Subplot 1
fig, ax = plt.subplots(2,2)
fig.suptitle('Image channel separation')
ax[0][0].imshow(img_cheeta)
ax[0][0].axis('off')
ax[0][0].set_title('RGB')
ax[0][1].imshow(r_channel, cmap='gray')
ax[0][1].axis('off')
ax[0][1].set_title('Red')
ax[1][0].imshow(g_channel, cmap='gray')
ax[1][0].axis('off')
ax[1][0].set_title('Green')
ax[1][1].imshow(b_channel, cmap='gray')
ax[1][1].axis('off')
ax[1][1].set_title('Blue')
fig.tight_layout()
plt.savefig('Intro_ImageAnalisis/Figure1')
plt.show()

#Gray scaled image
grayimg_cheeta = rgb2gray(img_cheeta)
# plt.plot(grayimg_jirafas)
# plt.show()

#Image histogram
hist_color = img_cheeta.flatten()
hist_gris = grayimg_cheeta.flatten()

#Subplot 2
input('Press enter to continue... ')
fig1, ax1 = plt.subplots(2,2,)
ax1[0][0].imshow(img_cheeta)
ax1[0][0].axis('off')
ax1[0][0].set_title('RGB')
ax1[0][1].hist(hist_color,bins=255)
ax1[0][1].set_title('RGB image histogram')
ax1[1][0].imshow(grayimg_cheeta, cmap='gray')
ax1[1][0].axis('off')
ax1[1][0].set_title('Gray-scale image')
ax1[1][1].hist(hist_gris,bins=255)
ax1[1][1].set_title('Histogram')
fig1.tight_layout()

plt.savefig('Intro_ImageAnalisis/Figure2')
plt.show()