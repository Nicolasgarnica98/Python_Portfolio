#Importacion de librerias
import numpy as np 
import requests
import skimage.io as io
from skimage.color import rgb2gray
from matplotlib import pyplot  as plt
import os
from skimage.filters import threshold_otsu
import nibabel 
import glob


#PARTE TEORICA ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
bina_otsu = img >= umbral

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


#PARTE BIOMÉDICA ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Carga de los datos
lista=glob.glob(os.path.join('Heart_Data', 'Data', '*.nii.gz'))
arch=nibabel.load(os.path.join('Heart_Data', 'Data', '4.nii.gz'))
print(arch.header['intent_name'])# El paciente al que corresponde
print(arch.header['descrip'])#número del corte actual 
print(arch.shape)#Revisar/resolucion
print(arch.header['slice_end'])#Cantidad de cortes paciente

vol1=np.empty([512,512,38],dtype=np.single)#Paciente 3
vol2=np.zeros([512,512,35],dtype=np.single)#Paciente 12
vol3=np.zeros([512,512,45],dtype=np.single)#Paciente 14

for i in lista:
    a=nibabel.load(os.path.join(i))
    numPa=int(str(a.header['intent_name']).split()[1][:-1])
    numCo=int(str(a.header['descrip']).split()[1][:-1])
     
    if numPa==3:
        vol1[:,:,numCo]=a.get_fdata()
    else:
        if numPa==12:
            vol2[:,:,numCo]=a.get_fdata()
        else:
            vol3[:,:,numCo]=a.get_fdata()    

plt.ion()
plt.show()
for i in range(vol1.shape[2]-1): #Se visualizan los cortes del paciente 3
    
    plt.imshow(vol1[:,:,i],cmap='gray')
    plt.title(f' Paciente 3 resonancia magnética cardiovascular corte {i}')
    plt.axis('off')
    plt.draw()
    plt.pause(0.000001)
    plt.clf()
    
plt.ion()
plt.show()
input("Press Enter to continue...")
for i in range(vol2.shape[2]-1):#Se visualizan los cortes del paciente 12
   
    plt.imshow(vol2[:,:,i],cmap='gray')
    plt.title(f' Paciente 12 resonancia magnética cardiovascular corte {i}')
    plt.axis('off')
    plt.draw()
    plt.pause(0.000001)
    plt.clf()
    
plt.ion()
plt.show()
input("Press Enter to continue...")
for i in range(vol3.shape[2]-1):#Se visualizan los cortes del paciente 14
    
    plt.imshow(vol3[:,:,i],cmap='gray')
    plt.title(f' Paciente 14 resonancia magnética cardiovascular corte {i}')
    plt.axis('off')
    plt.draw()
    plt.pause(0.000001)
    plt.clf()