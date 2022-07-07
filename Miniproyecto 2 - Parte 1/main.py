import os
import glob
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.exposure import match_histograms, equalize_hist
import scipy.signal as ss
from sklearn.metrics import mean_squared_error 
import cv2 as cv

v = [[1,2,3,4],[1,2,3,1],[1,4,5,3]]
mag = np.linalg.norm(v)
print(mag)

# image1=io.imread(os.path.join('roses.jpg'),as_gray=True)
# image2=io.imread(os.path.join('noisy_roses.jpg'),as_gray=True)

# def gaussian_kernel(size, sigma=1):#REFERENCIAR CODIGO
#     size = int(size) // 2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1 / (2.0 * np.pi * sigma**2)
#     g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
#     return g

# Kernel3a=np.array([[1,1,1],[1,1,1],[1,1,1]])
# Kernel3b=np.array([[1,1,1],[1,1,1],[1,1,1]])*(1/9)
# Kernel3c=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
# Kernel3d=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
# matriz1=np.array([[1,1,1],[1,1,1],[1,1,1]])
# def MyCCorrelation_201715400_201713127(image, kernel, boundary_condition):
    
#     alto=int(((kernel.shape[0]-1)/2))
#     ancho=int(((kernel.shape[1]-1)/2))
#     if(boundary_condition=='fill'):
#         img=np.pad(image,(alto,ancho),mode='constant', constant_values=0)
#         imagenSalida=np.zeros(shape=image.shape) #matriz de ceros de la imagen de salida
#         #Se realizan dos recorridos para recorrer cada fila y columna respectiva
#         for i in range(0,image.shape[1],1):
            
#             for j in range(0,image.shape[0],1):
#              imagenSalida[j,i]=sum(sum(img[0+j:kernel.shape[1]+j,0+i:kernel.shape[0]+i]*kernel))

#         CCorrelation=imagenSalida
#         return CCorrelation
#     if(boundary_condition=='symm'):
#         img=np.pad(image,(alto,ancho),'symmetric')
#         imagenSalida=np.zeros(shape=image.shape) #matriz de ceros de la imagen de salida
#         #Se realizan dos recorridos para recorrer cada fila y columna respectiva
#         for i in range(0,image.shape[1],1):
#             for j in range(0,image.shape[0],1):
#              imagenSalida[j,i]=sum(sum(img[0+j:kernel.shape[1]+j,0+i:kernel.shape[0]+i]*kernel))

#         CCorrelation=imagenSalida
#         return CCorrelation
    
#     if boundary_condition=='valid':#no toma las fronteras; toma los pixeles de adentro de mi imagen.
#         img=image
        
#         imagenSalida=np.zeros((img.shape[0]-int((kernel.shape[0]-1)/2)*2,img.shape[1]-int((kernel.shape[1]-1)/2)*2)) #matriz de ceros de la imagen de salida
#         #Se realizan dos recorridos para recorrer cada fila y columna respectiva
        
#         for i in range(0,imagenSalida.shape[1],1):
#             for j in range(0,imagenSalida.shape[0],1):
#                  imagenSalida[j,i]=sum(sum(img[0+j:kernel.shape[1]+j,0+i:kernel.shape[0]+i]*kernel))
        
        
#         return(imagenSalida)
    

# plt.figure(figsize = (20,10))
# plt.subplot(131)
# plt.imshow(image1, cmap="gray")
# plt.title('Imagen Original', fontsize=20)
# plt.axis('off')

# plt.subplot(132)
# plt.imshow(MyCCorrelation_201715400_201713127(image1, Kernel3a, 'fill'), cmap="gray")
# plt.title('My Cross-Correlación', fontsize=20)
# plt.axis('off')

# plt.subplot(133)
# plt.imshow(ss.correlate2d(image1, Kernel3a,boundary='fill'), cmap="gray")
# plt.title(' Función correlate2d', fontsize=20)
# plt.axis('off')
# input("Press Enter to continue...")
# plt.savefig('SubplotComparacionMetodos.jpg')
# plt.show()
# print(np.max(MyCCorrelation_201715400_201713127(image2, Kernel3a, 'fill')))
# print(np.max(MyCCorrelation_201715400_201713127(image2, Kernel3b, 'fill')))
# image1_vec=np.hstack(image1)
# image1_2d_vec=np.hstack(ss.correlate2d(image1, Kernel3a,boundary='fill'))
# image1_myc_vec=np.hstack(MyCCorrelation_201715400_201713127(image1, Kernel3a, 'fill'))
# # Calculation of Mean Squared Error (MSE) 
# error1=mean_squared_error(image1_vec,image1_myc_vec)
# error2=mean_squared_error(image1_vec,image1_2d_vec[:77120])
# print('MSE MyCCorrelation: ',error1)
# print('MSE Correlate2d: ',error2)
# # PUNTO CROSS-CORRELACION

# plt.figure(figsize = (20,10))
# plt.subplot(131)
# plt.imshow(image2, cmap="gray")
# plt.title('Imagen Original', fontsize=20)
# plt.axis('off')

# plt.subplot(132)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, Kernel3a, 'fill'), cmap="gray")
# plt.title('My Cross-Correlación Kernel 3a', fontsize=18)
# plt.axis('off')

# plt.subplot(133)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, Kernel3b, 'fill'), cmap="gray")
# plt.title('My Cross-Correlación Kernel 3b', fontsize=18)
# plt.axis('off')
# input("Press Enter to continue...")
# plt.savefig('SubplotComparacionKernels3a3b.jpg')
# plt.show()
# #COMPARACION kernel Gaussiano de tamaño 5x5 y sigma = 1 Y KERNEL 3b
# plt.figure(figsize = (20,10))
# plt.subplot(131)
# plt.imshow(image2, cmap="gray")
# plt.title('Imagen Original', fontsize=20)
# plt.axis('off')

# plt.subplot(132)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, Kernel3b, 'fill'), cmap="gray")
# plt.title('My Cross-Correlación Kernel 3b', fontsize=18)
# plt.axis('off')

# plt.subplot(133)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, gaussian_kernel(5, sigma=1), 'fill'), cmap="gray")
# plt.title('My Cross-Correlación 5x5', fontsize=18)
# plt.axis('off')
# input("Press Enter to continue...")
# plt.savefig('SubplotComparacionKernel5x5.jpg')
# plt.show()
# #Comparacion de Kernels con distintos sigmas
# plt.figure(figsize = (20,10))
# plt.subplot(131)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, gaussian_kernel(5, sigma=3), 'fill'), cmap="gray")
# plt.title('My Cross-Correlación sigma=3', fontsize=18)
# plt.axis('off')

# plt.subplot(132)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, gaussian_kernel(5, sigma=6), 'fill'), cmap="gray")
# plt.title('My Cross-Correlación sigma=6', fontsize=18)
# plt.axis('off')

# plt.subplot(133)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, gaussian_kernel(5, sigma=9), 'fill'), cmap="gray")
# plt.title('My Cross-Correlación sigma=9', fontsize=18)
# plt.axis('off')
# input("Press Enter to continue...")
# plt.savefig('SubplotComparacionKernelsigmas.jpg')
# plt.show()
# #Comparacion de Kernels con distintos tamaños
# plt.figure(figsize = (20,10))
# plt.subplot(131)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, gaussian_kernel(3, sigma=1), 'fill'), cmap="gray")
# plt.title('My Cross-Correlación size=3', fontsize=18)
# plt.axis('off')

# plt.subplot(132)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, gaussian_kernel(5, sigma=1), 'fill'), cmap="gray")
# plt.title('My Cross-Correlación size=5', fontsize=18)
# plt.axis('off')

# plt.subplot(133)
# plt.imshow(MyCCorrelation_201715400_201713127(image2, gaussian_kernel(7, sigma=1), 'fill'), cmap="gray")
# plt.title('My Cross-Correlación size=7', fontsize=18)
# plt.axis('off')
# input("Press Enter to continue...")
# plt.savefig('SubplotComparacionKerneltamaños.jpg')
# plt.show()
# #Plicacion Filtros Kernels 3c y 3d
# plt.figure(figsize = (20,10))
# plt.subplot(122)
# plt.imshow(MyCCorrelation_201715400_201713127(image1, Kernel3c, 'fill'), cmap="gray")
# plt.title('My Cross-Correlación Kenel3c', fontsize=30)
# plt.axis('off')

# plt.subplot(121)
# plt.imshow(MyCCorrelation_201715400_201713127(image1, Kernel3d, 'fill'), cmap="gray")
# plt.title('My Cross-Correlación Kernel3d', fontsize=30)
# plt.axis('off')

# input("Press Enter to continue...")
# plt.savefig('SubplotComparacionKernel3c3d.jpg')
# plt.show()
# #BONO
# plt.figure(figsize = (20,10))
# plt.subplot(122)
# plt.imshow(MyCCorrelation_201715400_201713127(MyCCorrelation_201715400_201713127(image1, Kernel3c, 'fill'), Kernel3d, 'fill'), cmap="gray")
# plt.title('My Cross-Correlación Kenel3c&3d', fontsize=30)
# plt.axis('off')
# gdx=MyCCorrelation_201715400_201713127(image1, Kernel3c, 'fill')
# gdy=MyCCorrelation_201715400_201713127(image1, Kernel3d, 'fill')

# plt.subplot(121)
# plt.imshow((gdx**2+gdy**2)**0.5,cmap='gray')
# plt.title('My Cross-Correlación magnitud', fontsize=30)
# plt.axis('off')

# input("Press Enter to continue...")
# plt.savefig('SubplotComparacionKernelBONO.jpg')
# plt.show()




# #PROBLEMA BIOMÉDICO--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# #Carga de imágenes
# df = glob.glob(os.path.join('malaria_dataset/template','*.jpeg'))
# df_template = glob.glob(os.path.join('malaria_dataset/template','*.jpg'))
# df_template = df_template + df
# df_train = glob.glob(os.path.join('malaria_dataset/train','*.png'))
# #Función de carga de las imágenes
# def cargar_imagenes(df_imagenes):
#     img_array = []
#     for i in range(len(df_imagenes)):
#         img_array.append(rgb2gray(io.imread(df_imagenes[i])))
#     return img_array

# img_train = cargar_imagenes(df_train)
# img_template = cargar_imagenes(df_template)

# #Funcion de pre-procesamiento
# def myImagePreprocessor(image, target_hist, action):
#     #Ecualizacion de la imagn original y las imagenes de referencia
#     eq_image = equalize_hist(image)
#     eq_refimage = equalize_hist(target_hist)
#     #Matching del histograma de la imagen ecualizada con el histograma de la imagen de referencia ecualizada
#     matched_image = match_histograms(eq_image,eq_refimage)
    
#     #Subplot
#     fig, ax=plt.subplots(5,2)
#     ax[0][0].imshow(image, cmap='gray')
#     ax[0][0].axis('off')
#     ax[0][1].hist(image.flatten(), 256)
#     ax[1][0].imshow(eq_image, cmap='gray')
#     ax[1][0].axis('off')
#     ax[1][1].hist(eq_image.flatten(), 256)
#     ax[2][0].imshow(target_hist, cmap='gray')
#     ax[2][0].axis('off')
#     ax[2][1].hist(target_hist.flatten(), 256)
#     ax[3][0].imshow(eq_refimage, cmap='gray')
#     ax[3][0].axis('off')
#     ax[3][1].hist(eq_refimage.flatten(), 256)
#     ax[4][0].imshow(matched_image, cmap='gray')
#     ax[4][0].axis('off')
#     ax[4][1].hist(matched_image.flatten(), 256)
#     fig.tight_layout()
#     if action == 'save':
#         plt.savefig('Figura1.png')
#     elif action == 'show':
#         plt.show()


#     return matched_image

# myImagePreprocessor(img_train[0],img_template[1],'show')




