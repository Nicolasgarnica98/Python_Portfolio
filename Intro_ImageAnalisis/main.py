import numpy as np
import requests
import skimage.io as io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

#Descarga de la imagen seleccionada
r = requests.get('https://www.sciencemag.org/sites/default/files/styles/article_main_large/public/images/cc_Giraffes_16x9.jpg?itok=dKmuVKO6')
with open('img_jirafas.png', 'wb') as f:
      f.write(r.content)

#Carga de la imágen seleccionada
img_jirafas = io.imread(os.path.join(('img_jirafas.png')))
print(img_jirafas.shape)
#Obtencion de canales de la imagen

#Canal R
r_channel = img_jirafas.copy()
r_channel[:,:,1] = 0
r_channel[:,:,2] = 0
r_channel = rgb2gray(r_channel)

#Canal G
g_channel = img_jirafas.copy()
g_channel[:,:,0] = 0
g_channel[:,:,2] = 0
g_channel = rgb2gray(g_channel)

#Canal B
b_channel = img_jirafas.copy()
b_channel[:,:,0] = 0
b_channel[:,:,1] = 0
b_channel = rgb2gray(b_channel)


#Subplot 1
fig, ax = plt.subplots(2,2)
fig.suptitle('Canales de la imagen RGB')
ax[0][0].imshow(img_jirafas)
ax[0][0].axis('off')
ax[0][0].set_title('Imagen a color (RGB)')
ax[0][1].imshow(r_channel, cmap='gray')
ax[0][1].axis('off')
ax[0][1].set_title('Primer canal (R)')
ax[1][0].imshow(g_channel, cmap='gray')
ax[1][0].axis('off')
ax[1][0].set_title('Segundo canal (G)')
ax[1][1].imshow(b_channel, cmap='gray')
ax[1][1].axis('off')
ax[1][1].set_title('Tercer canal (B)')
fig.tight_layout()
plt.savefig('Figura 1')
plt.show()

#Conversion de la imagen original a grises
grayimg_jirafas = rgb2gray(img_jirafas)
# plt.plot(grayimg_jirafas)
# plt.show()
#Obtencion de los histogramas
hist_color = img_jirafas.flatten()
hist_gris = grayimg_jirafas.flatten()

#Subplot 2
input('Press enter to continue... ')
fig1, ax1 = plt.subplots(2,2,)
ax1[0][0].imshow(img_jirafas)
ax1[0][0].axis('off')
ax1[0][0].set_title('Imagen a color (RGB)')
ax1[0][1].hist(hist_color,bins=255)
ax1[0][1].set_title('Histograma imagen a color')
ax1[1][0].imshow(grayimg_jirafas, cmap='gray')
ax1[1][0].axis('off')
ax1[1][0].set_title('Imagen en grises')
ax1[1][1].hist(hist_gris,bins=255)
ax1[1][1].set_title('Histograma imagen en grises')
fig1.tight_layout()

plt.savefig('Figura 2')
plt.show()