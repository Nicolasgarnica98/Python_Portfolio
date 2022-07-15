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


#Thershold binarization ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Image download
r = requests.get("https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/11-Edge-Detection/Hough_Transform_Circles/coins.png")
with open('monedas', 'wb') as f:
    f.write(r.content)

#Image load
img=io.imread(os.path.join('monedas'))
flat_g=np.ndarray.flatten(img)# gray scaled image histogram
print(img.shape)#checking image dimensions

fig0 , ax0 = plt.subplots(1,2)
ax0[0].imshow(img ,cmap='gray')
ax0[0].set_title('Image')
ax0[0].axis('off')
ax0[1].hist(flat_g, bins=60, range=(0, 255),facecolor='blue',alpha=0.75, density=False)
ax0[1].set_title('Image histogram')
ax0[1].grid(True)
fig0.savefig('subplot1.png')
fig0.tight_layout()
fig0.show()



#Otsu method for umbralization
umbral = threshold_otsu(img)#106 is the threshold obtained
bina_otsu = img >= umbral #applying binarization mask to image

#binarization using the 60 percentile
per = np.percentile(img, 60)#threshold was 39
bina_per= img >= per #applying binarization mask to image

#Image binarization with threshold = 75
bina_175=img >= 175

#Binarization with chosen thresholds
umbral_sup=250
umbral_inf=60

bina_esc=np.zeros((img.shape[0],img.shape[1]))

#Binarization with two thresholds -> mask generation
for f in range (0, img.shape[0], 1 ):
        for j in range (0, img.shape[1], 1 ):

            if img[f,j]>= umbral_inf and img[f,j]<=umbral_sup:
                bina_esc[f,j]=True
            else:
                bina_esc[f,j]=False

input("Press Enter to continue...")


fig, ax = plt.subplots(2,4)
ax[0][0].set_title('Máscara 1\n(Otsu)', wrap=True, fontsize=10)
ax[0][0].imshow(bina_otsu, cmap='gray')
ax[0][0].axis('off')
ax[0][1].set_title('Máscara 2\n(percentil)', wrap=True, fontsize=10)
ax[0][1].imshow(bina_per, cmap='gray')
ax[0][1].axis('off')
ax[0][2].set_title('Máscara 3\n(Umb. aleatorio)', wrap=True, fontsize=10)
ax[0][2].imshow(bina_175, cmap='gray')
ax[0][2].axis('off')
ax[0][3].set_title('Máscara 4\n(Umb. arbitrario)', wrap=True, fontsize=10)
ax[0][3].imshow(bina_esc, cmap='gray')
ax[0][3].axis('off')
ax[1][0].set_title('Segmentacion\nOtsu', wrap=True, fontsize=10)
ax[1][0].imshow(bina_otsu*img, cmap='gray')
ax[1][0].axis('off')
ax[1][1].set_title('Segmentación\npercentil',wrap=True, fontsize=10)
ax[1][1].imshow(bina_per*img, cmap='gray')
ax[1][1].axis('off')
ax[1][2].set_title('Segmentación\numb. aleatorio', wrap=True, fontsize=10)
ax[1][2].imshow(bina_175*img, cmap='gray')
ax[1][2].axis('off')
ax[1][3].set_title('Segmentación\numb.\narbitrario', wrap=True, fontsize=10)
ax[1][3].imshow(bina_esc*img, cmap='gray')
ax[1][3].axis('off')
fig.savefig('subplotLAB.png')
fig.tight_layout()
plt.show()



#Tomography visualization ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Data load
lista=glob.glob(os.path.join('Heart_Data', 'Data', '*.nii.gz'))
arch=nibabel.load(os.path.join('Heart_Data', 'Data', '4.nii.gz'))
print(arch.header['intent_name'])# Pacient
print(arch.header['descrip'])#Number of the slice 
print(arch.shape)#Image resolution
print(arch.header['slice_end'])#Number of slices per patient

vol1=np.empty([512,512,38],dtype=np.single)#Patient 3
vol2=np.zeros([512,512,35],dtype=np.single)#Patient 12
vol3=np.zeros([512,512,45],dtype=np.single)#Patient 14

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