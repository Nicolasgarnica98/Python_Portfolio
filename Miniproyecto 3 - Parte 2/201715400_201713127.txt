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
from skimage.morphology import dilation, erosion, h_minima, selem
from skimage.segmentation import watershed
from prettytable import PrettyTable
from sklearn.metrics import jaccard_score
from skimage.filters import threshold_otsu

fruits= io.imread(os.path.join('data_mp3', 'fruits_binary.png'))
fruits=fruits>=50
fruits=fruits.astype(int)
#bin_img=fruits1


def MyConnComp_201715400_201713127(binary_image,conn):
    bin_img_e=np.pad(binary_image,(1,1),mode='constant', constant_values=0)
    
    labeled_image=np.zeros((bin_img_e.shape[0],bin_img_e.shape[1]))
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])

    if conn==4:
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    if conn==8:
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    idx=1
    pixel_labels=[]
    for i in range(0,binary_image.shape[0]):
        for j in range(0,binary_image.shape[1]):
            if bin_img_e[i+1,j+1]==1:
                H1=np.zeros((bin_img_e.shape[0],bin_img_e.shape[1]))
                H1[i+1,j+1]=bin_img_e[i+1,j+1]
                
                end=False
                while (end==False):
                    temp=H1
                    H1=ndi.binary_dilation(H1,structure=kernel)
                    H1=np.logical_and(H1,bin_img_e) 
                    if np.array_equal(H1,temp):
                        end=True
                        
                bin_img_e=bin_img_e-H1     
                H1=H1*idx
                labeled_image=labeled_image+H1
                print('Elemento conexo ', idx,' listo: ')
                idx=idx+1 
    labeled_image=labeled_image[1:labeled_image.shape[0]-1,1:labeled_image.shape[1]-1]   
    for i in range(1,int(np.max(labeled_image))+1):
        pixel_labels.append(np.ravel_multi_index(np.where(labeled_image==i),labeled_image.shape) )
    return(labeled_image,pixel_labels)


labeled_image,pixel_labels=MyConnComp_201715400_201713127(fruits,8)

input("Figura 1. Press Enter to continue...")
plt.imshow(fruits,cmap='gray')    
plt.show()

plt.imshow(labeled_image,cmap='gray')    
plt.show()

plt.figure(figsize = (20,10))
plt.subplot(121)
plt.imshow(fruits, cmap="gray")
plt.title('Imagen Original', fontsize=50)
plt.axis('off')

plt.subplot(122)
plt.imshow(labeled_image, cmap="gray")
plt.title('Labeled Image', fontsize=50)
plt.axis('off')

plt.tight_layout()
plt.savefig('SubplotComparacion.jpg')
plt.show()


#Imagen 20x20 donde 4-conectividad y 8-conectividad son iguales
binary_image1=np.array([
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  ])
#aplicacion de la función
binary_image1_c4,pixel_labels1_c4=MyConnComp_201715400_201713127(binary_image1,4)
binary_image1_c8,pixel_labels1_c8=MyConnComp_201715400_201713127(binary_image1,8)
#subplot

input("Figura 2. Press Enter to continue...")
plt.figure(figsize = (40,20))
plt.subplot(131)
plt.imshow(binary_image1, cmap="gray")
plt.title('Imagen Original', fontsize=50)
plt.axis('off')

plt.subplot(132)
plt.imshow(binary_image1_c4, cmap="gray")
plt.title('labeled image \n1 - 4 conectividad', fontsize=50)
plt.axis('off')

plt.subplot(133)
plt.imshow(binary_image1_c8, cmap="gray")
plt.title('labeled image \n1 - 8 conectividad', fontsize=50)
plt.axis('off')

plt.tight_layout()
plt.savefig('IMG_iguales.jpg')
plt.show()


#Imagen 20x20 donde 4-conectividad y 8-conectividad son diferentes
binary_image2=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  ])
#aplicacion de la función
binary_image2_c4,pixel_labels2_c4=MyConnComp_201715400_201713127(binary_image2,4)
binary_image2_c8,pixel_labels2_c8=MyConnComp_201715400_201713127(binary_image2,8)


#subplot
input("Figura 3. Press Enter to continue...")

plt.figure(figsize = (40,20))
plt.subplot(131)
plt.imshow(binary_image2, cmap="gray")
plt.title('Imagen Original', fontsize=50)
plt.axis('off')


plt.subplot(132)
plt.imshow(binary_image2_c4, cmap="gray")
plt.title('labeled image \n1 - 4 conectividad', fontsize=50)
plt.axis('off')

plt.subplot(133)
plt.imshow(binary_image2_c8, cmap="gray")
plt.title('labeled image \n1 - 8 conectividad', fontsize=50)
plt.axis('off')

plt.tight_layout()
plt.savefig('IMG_diferentes.jpg')
plt.show()



#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#PUNTO 2
imagen_prueba = os.path.join('data_mp3','star_binary.png')
lista=glob.glob(os.path.join('data_mp3','blood_cell_dataset', 'noisy_data', '*.png'))
gt=glob.glob(os.path.join('data_mp3','blood_cell_dataset', 'groundtruth', '*.png'))
lista_b=[]
lista_i = []
kernel = np.ones((5,5),np.float32)/25
#https://code.tutsplus.com/es/tutorials/image-filtering-in-python--cms-29202 LINK USADO

#Organizacion de datos de 1 a 10
df_gt = []
df_img = []
for i in range(0,len(gt)):
    df_gt.append('1')
    df_img.append('1')

for i in range(0,len(gt)):
    index_gt = int([float(index_gt) for index_gt in re.findall(r'-?\d+\.?\d*', gt[i][8:])][0])-1
    index_lista = int([float(index_lista) for index_lista in re.findall(r'-?\d+\.?\d*', lista[i][8:])][0])-1
    df_img[index_lista] = lista[i]
    df_gt[index_gt] = gt[i]

#Referencia: https://blog.finxter.com/how-to-extract-numbers-from-a-string-in-python/

#Carga imágenes
for i in range (0,len(df_img)):
    lista_i.append(cv2.imread(df_img[i]))
    lista_b.append(cv2.imread(df_img[i]))

img_prueba = lista_b[8]

#Función de watersheds sin marcadores
def get_watersheds_nm(img):
    pp_img = cv2.medianBlur(img, 5)
    pp_img = rgb2gray(pp_img)
    gradient = dilation(pp_img)-erosion(pp_img)
    ws = watershed(gradient,watershed_line=True)
    return ws, gradient

ws_img_prueba_nm, grad_img_prueba = get_watersheds_nm(img_prueba)

#Subplot 1
input('Figura 4. Press enter to continue...')
fig3, ax3 = plt.subplots(1,3)
ax3[0].set_title('Imagen original')
ax3[0].imshow(img_prueba)
ax3[0].axis('off')
ax3[1].set_title('Gradiente')
ax3[1].imshow(grad_img_prueba,'gray')
ax3[1].axis('off')
ax3[2].set_title('Watersheds \nsin maracdor')
ax3[2].imshow(ws_img_prueba_nm,'gray')
ax3[2].axis('off')
fig3.tight_layout()
fig3.savefig('Watersheds_nomarker.png')
plt.show()

#Función de watersheds con marcadores
def get_watersheds(img):
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    pp_img = cv2.medianBlur(img, 5)
    gray_img = np.uint8(rgb2gray(pp_img)*255)
    gradient = dilation(gray_img)-erosion(gray_img)
    gradient = gradient.astype(np.float)
    minimos = h_minima(gray_img, 65, selem=kernel)
    dil_cc = MyConnComp_201715400_201713127(minimos,8)[0]
    marcadores = dilation(dil_cc,selem=kernel)
    ws = watershed(gradient, markers=marcadores)
    return gray_img, gradient, marcadores, ws

gray_img, gradiente, marcadores, seg_ws = get_watersheds(img_prueba)

#Subplot 2
input('Figura 5. Press enter to continue...')
fig4, ax4 = plt.subplots(1,4)
ax4[0].set_title('Imagen en \nescala de grises')
ax4[0].imshow(gray_img,'gray')
ax4[0].axis('off')
ax4[1].set_title('Gradiente \nmorfológico')
ax4[1].imshow(gradiente,'gray')
ax4[1].axis('off')
ax4[2].set_title('Marcadores')
ax4[2].imshow(marcadores,'gray')
ax4[2].axis('off')
ax4[3].set_title('Segmentación con \nwatersheds y marcadores')
ax4[3].imshow(seg_ws,'gray')
ax4[3].axis('off')
plt.tight_layout(pad=0.4, w_pad=1.2)
fig4.savefig('Watersheds.png')
plt.show()

#Función binarización, separación del fondo, utilizando OTSU:
def get_mascara_binaria(img_seg):
    img_seg = img_seg/np.max(img_seg)
    umbral = threshold_otsu(img_seg)
    masc_bin = img_seg >= umbral
    respuesta = masc_bin
    if np.count_nonzero(masc_bin.flatten())> np.count_nonzero(masc_bin.flatten() == 0):
        print('verdad')
        masc_bin_mod = np.zeros((masc_bin.shape[0],masc_bin.shape[1]))
        for i in range(0,masc_bin.shape[0]):
            for j in range(0,masc_bin.shape[1]):
                if masc_bin[i][j] == False:
                    masc_bin_mod[i][j] = 1
                elif masc_bin[i][j] == True:
                    masc_bin_mod[i][j] == 0
        respuesta =  masc_bin_mod                   

    return respuesta

#Obtención de watersheds y máscaras binarias
masc_wsnm_img = []
masc_ws_img = []
for i in range(0,len(lista_b)):
    masc_ws_img.append(get_mascara_binaria(get_watersheds(lista_b[i])[3]))
    masc_wsnm_img.append(get_mascara_binaria(get_watersheds_nm(lista_b[i])[0]))

#Craga de las imágenes de ground truth
img_gt = []
for i in range(0,len(gt)):
    img_gt.append(cv2.imread(df_gt[i]))

#Binarización imágenes de Ground Truth
for i in range(0,len(img_gt)):
    img_gt[i] = rgb2gray(img_gt[i])
    umbral = 1
    img_gt[i] = img_gt[i] >= umbral

prueba = get_mascara_binaria(seg_ws)
comp_gt = img_gt[8]

input('Figura 6. Press enter to continue...')
fig5, ax5=plt.subplots(1,2)
ax5[0].set_title('Segmentación de \nimagen 9 de \nGround Truth')
ax5[0].imshow(comp_gt,'gray')
ax5[0].axis('off')
ax5[1].set_title('Segmentación de \nimagen 9 con \nwatersheds y OTSU')
ax5[1].imshow(prueba,'gray')
ax5[1].axis('off')
fig5.tight_layout()
plt.show()
fig5.savefig('comparacion_seg.png')

#Obtención del índice de Jaccard para los tres métodos
jaccard_m3 = [0.812113772961991, 0.7394182012276488, 0.7933115413628363, 0.7630499248486818, 0.7383991293163155, 0.8205280172413794,
 0.8257937772114261, 0.7601846182828615, 0.7444213031835764, 0.7024377856780092]#Resultados del índice de Jaccard para imágenes con pre-procesamiento de la entrega anterior.
jaccard_ws = []
jaccard_wsnm = []

for i in range(len(img_gt)):
    jaccard_ws.append(jaccard_score(img_gt[i].flatten(),masc_ws_img[i].flatten()))
    jaccard_wsnm.append(jaccard_score(img_gt[i].flatten(),masc_wsnm_img[i].flatten()))

jaccard_ws = np.around(jaccard_ws,decimals=4)
jaccard_wsnm = np.around(jaccard_wsnm,decimals=4)
jaccard_m3 = np.around(jaccard_m3,decimals=4)

#Tabla de resultados
tabla_resultados = PrettyTable(['Imagen','Watersheds + marcador','Watersheds','Relleno huecos (Pre-procesamiento)'])
for i in range(0,len(lista_b)+2):
    if i < len(lista_b) :
        tabla_resultados.add_row([str(i+1),str(jaccard_ws[i]),str(jaccard_wsnm[i]),str(jaccard_m3[i])])
    elif i == len(lista_b):
        tabla_resultados.add_row(['Promedio',str(np.round(np.mean(jaccard_ws),4)),str(np.round(np.mean(jaccard_wsnm),4)),str(np.round(np.mean(jaccard_m3),4))])
    else:
        tabla_resultados.add_row(['Desviación',str(np.round(np.std(jaccard_ws),4)),str(np.round(np.std(jaccard_wsnm),4)),str(np.round(np.std(jaccard_m3),4))])

print(tabla_resultados)