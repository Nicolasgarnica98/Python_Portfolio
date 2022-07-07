#Importacion de librerias
import numpy as np 
import requests
import skimage.io as io
from skimage.color import rgb2gray
from matplotlib import pyplot  as plt
import os
from skimage.filters import threshold_otsu
#Se descarga la imagen de internet 
r = requests.get("https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/11-Edge-Detection/Hough_Transform_Circles/coins.png")
with open('monedas', 'wb') as f:
    f.write(r.content)

#Se carga la imagen obtenida de internet
img=io.imread(os.path.join('monedas'))
flat_g=np.ndarray.flatten(img)# Se realiza el histograma de la imagen a escala de grises
print(img.shape)#Dimensiones de la imagen

fig0 , ax0 = plt.subplots(1,2)
ax0[0].imshow(img ,cmap='gray')# Se grafica la imagen descargada
ax0[0].set_title('Monedas')
ax0[0].axis('off')
ax0[1].hist(flat_g, bins=60, range=(0, 255),facecolor='blue',alpha=0.75, density=False)
ax0[1].set_title('Histograma de Imagen Monedas')
ax0[1].grid(True)
fig0.savefig('subplot1.png')
fig0.tight_layout()
fig0.show()



#Metodo de Otsu
umbral = threshold_otsu(img)#Umbral fue de 106 (Máscara)
bina_otsu = img <= umbral

num = 0
for i in range(0,len(img.flatten())):
    if img.flatten()[i] == umbral:
        num = i
        break
print(num)
print(int(bina_otsu.flatten()[num]))


#Metodo de binarización con el percentil 60 de las intensidades (Máscara)
per = np.percentile(img, 60)#Umbral fue de 39
bina_per= img >= per

# Binarización con un umbral de 175 (Máscara)
bina_175=img >= 175

#Binarizacion Umbrales escogidos (Máscara)
umbral_sup=250
umbral_inf=60

bina_esc=np.zeros((img.shape[0],img.shape[1]))

#Binarización de umbrales dobles (Máscara)
for f in range (0, img.shape[0], 1 ):
        for j in range (0, img.shape[1], 1 ):

            if img[f,j]>= umbral_inf and img[f,j]<=umbral_sup:
                bina_esc[f,j]=True
            else:
                bina_esc[f,j]=False

input("Press Enter to continue...")

fig, ax = plt.subplots(2,4)
ax[0][0].set_title('Máscara 1\n(Otsu)', wrap=True)
ax[0][0].imshow(bina_otsu, cmap='gray')
ax[0][0].axis('off')
ax[0][1].set_title('Máscara 2\n(percentil)', wrap=True)
ax[0][1].imshow(bina_per, cmap='gray')
ax[0][1].axis('off')
ax[0][2].set_title('Máscara 3\n(Umbral aleatorio)', wrap=True)
ax[0][2].imshow(bina_175, cmap='gray')
ax[0][2].axis('off')
ax[0][3].set_title('Máscara 4\n(Umbrales arbitrarios)', wrap=True)
ax[0][3].imshow(bina_esc, cmap='gray')
ax[0][3].axis('off')
ax[1][0].set_title('Segmentacion\nOtsu', wrap=True)
ax[1][0].imshow(bina_otsu*img, cmap='gray')
ax[1][0].axis('off')
ax[1][1].set_title('Segmentación\npercentil',wrap=True)
ax[1][1].imshow(bina_per*img, cmap='gray')
ax[1][1].axis('off')
ax[1][2].set_title('Segmentación\numbral aleatorio', wrap=True)
ax[1][2].imshow(bina_175*img, cmap='gray')
ax[1][2].axis('off')
ax[1][3].set_title('Segmentación\numbrales\narbitrarios', wrap=True)
ax[1][3].imshow(bina_esc*img, cmap='gray')
ax[1][3].axis('off')
fig.savefig('subplotLAB.png')
fig.tight_layout()
plt.show()












































