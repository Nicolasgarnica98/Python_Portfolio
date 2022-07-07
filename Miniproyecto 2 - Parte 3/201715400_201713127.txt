import os
import csv
import glob
import numpy as np
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import confusion_matrix
from skimage.color import rgb2hsv, rgb2lab

#Craga de imagen y gropund truth
mat = scipy.io.loadmat('flower_classifier_results.mat')
gt=mat["groundtruth"]
pred=mat["predictions"]

#Funcion de matriz de confusión
def MyConfMatrix_201715400_201713127(gt, pred):
    gt_1={}
    pred_1={}
    for i in range(0,len(gt),1):
        gt_1[i]=gt[i]
        pred_1[i]=pred[i]
    gt=list(gt_1.values())
    pred=list(pred_1.values())
    Unique=np.unique(np.array(gt+pred))
    Unique_num={}
    cont=0
    for i in Unique:
        Unique_num[i]=cont
        cont+=1
    #TRANSFORMACION
    gt_t= np.zeros((len(gt)))
    pred_t= np.zeros((len(pred)))     
    for i in range(0, len(gt),1):
        gt_t[i]=Unique_num[gt[i]] 
        pred_t[i]=Unique_num[pred[i]] 
    conf_matrix= np.zeros((len(Unique_num), len(Unique_num)))           
    for i in range (0,len(gt),1):
        conf_matrix[int(gt_t[i]),int(pred_t[i])]= conf_matrix[int(gt_t[i]),int(pred_t[i])]+1
    ##TRue positives
    TP=np.zeros((len(Unique_num)))
    for i in range(0,len(Unique_num),1):
        TP[i]=conf_matrix[i,i]
    FP = -np.diag(conf_matrix)+conf_matrix.sum(axis=0)##False positives
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)##False begatives
    TN = sum(sum(conf_matrix)) -FP-FN-TP ##True Negative
    prec_class_1= np.round_(TP/(TP+FP), decimals=3)
    rec_class_1= np.round_(TP/(TP+FN), decimals=3)
    mean_prec=np.mean(prec_class_1)
    mean_rec=np.mean(rec_class_1)
    prec_class={}
    rec_class={}
    for i in Unique:
        prec_class[i]=prec_class_1[Unique_num[i]]
        rec_class[i]=rec_class_1[Unique_num[i]]
    
    f1 = (2*mean_prec*mean_rec)/(mean_rec+mean_prec)
    
    return conf_matrix,prec_class,rec_class,mean_prec, mean_rec, f1

conf_matrix,prec_class,rec_class,mean_prec, mean_rec=MyConfMatrix_201715400_201713127(gt, pred)


#PARTE BIOMÉDICA ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Carga de imágenes
df = glob.glob(os.path.join('malaria_dataset/template','*.jpeg'))
df_template = glob.glob(os.path.join('malaria_dataset/template','*.jpg'))
df_template = df_template + df
df_train = glob.glob(os.path.join('malaria_dataset/train','*.png'))
df_test = glob.glob(os.path.join('malaria_dataset/test','*.png'))
df_test = [df_test[0],df_test[2],df_test[3],df_test[4],df_test[5],df_test[6],df_test[7],df_test[8],df_test[9],df_test[1]]
#Función de carga de las imágenes
def cargar_imagenes(df_imagenes):
    img_array = []
    for i in range(len(df_imagenes)):
        img_array.append(io.imread(df_imagenes[i]))
    return img_array

img_train = cargar_imagenes(df_train)
img_template = cargar_imagenes(df_template)
img_test = cargar_imagenes(df_test)

#Funcion de histograma concatenado de color
def MyColorHist_201715400_201713127(color_image, space, plot):
    img_color = color_image
    if space == 'HSV':
        img_color = rgb2hsv(img_color)
    elif space == 'Lab':
        img_color = rgb2lab(img_color)
    #Concatenar los nuevos rangos para hallar el histograma
    def concatenado(imgacolor):

        hist_c1 = imgacolor[:,:,0].flatten()
        hist_c2 = imgacolor[:,:,1].flatten()
        hist_c3 = imgacolor[:,:,2].flatten()
        Hist = []
        
        #Verificar que los valores negativos entren en el rango que les corresponderá
        if min(hist_c1)<0:
            for i in range(0,len(hist_c1)):
                Hist.append(hist_c1[i] + min(hist_c1))
        else:
            for i in range(0,len(hist_c1)):
                Hist.append(hist_c1[i])
        if min(hist_c2)<0:
            for i in range (0,len(hist_c2)):
                Hist.append(hist_c2[i] + 256 + min(hist_c2))
        else:
            for i in range (0,len(hist_c2)):
                Hist.append(hist_c2[i] + 256)
        if min(hist_c3)<0:
            for i in range(0,len(hist_c3)):
                Hist.append(hist_c3[i]+513+min(hist_c3))
        else:
            for i in range(0,len(hist_c3)):
                Hist.append(hist_c3[i]+513)
        return Hist
    
    #Creación del histograma, obteniendo las distribuciones de probabilidad
    prob, bins, hist = plt.hist(concatenado(img_color),768,density=True)
    for item in hist:
        item.set_height(item.get_height()/sum(prob))

    conc_hist = prob

    #Plot opcional
    if plot ==True:
        fig, ax=plt.subplots(5)
        fig.suptitle('Canales e histogramas de la imagen')
        ax[0].set_title('Imagen en espacio de color '+space)
        ax[0].imshow(img_color)
        ax[0].axis('off')
        ax[1].set_title('Canal '+space[0])
        ax[1].imshow(img_color[:,:,0], cmap='gray')
        ax[1].axis('off')
        ax[2].set_title('Canal '+space[1])
        ax[2].imshow(img_color[:,:,1], cmap='gray')
        ax[2].axis('off')
        ax[3].set_title('Canal '+space[2])
        ax[3].imshow(img_color[:,:,2], cmap='gray')
        ax[3].axis('off')
        ax[4].set_title('Histograma concatenado')
        ax[4].hist(concatenado(img_color),768, density=True)
        fig.tight_layout()
        plt.show()

    return conc_hist

print('Creando el plot...')
MyColorHist_201715400_201713127(img_train[0], 'RGB',True)

#Cálculo de histogramoas molde. Se demora un poquito!!
print('Calculando histogramas molde...')
htrain_infected_RGB = MyColorHist_201715400_201713127(img_train[0], 'RGB',False)
htrain_uninfected_RGB = MyColorHist_201715400_201713127(img_train[1], 'RGB',False)
htrain_infected_Lab = MyColorHist_201715400_201713127(img_train[0], 'Lab',False)
htrain_uninfected_Lab = MyColorHist_201715400_201713127(img_train[1], 'Lab',False)
htrain_infected_HSV = MyColorHist_201715400_201713127(img_train[0], 'HSV',False)
htrain_uninfected_HSV = MyColorHist_201715400_201713127(img_train[1], 'HSV',False)

#Función de kernel de intersección
def MyIntersectionKernel_201715400_201713127(hist1, hist2):
    minimos_array = []
    #hallar los minimos para cada numero entre los histogramas a comparar
    for i in range(0,len(hist1)):
        if hist1[i]<hist2[i]:
            minimos_array.append(hist1[i])
        else:
            minimos_array.append(hist2[i])
    #Hallar la intersccion sumando los minimos y dividiento 
    interseccion = np.sum(minimos_array)/np.sum(hist2)

    return interseccion

#Función bono de distancia Chi^2
def MyChi2Distance_201715400_201713127(hist1, hist2):
    resta = []
    for i in range(0,len(hist1)):
        resta.append((hist1[i]-hist2[i])**2)
    dc2 = np.sum(resta)/np.sum(hist1)

    return dc2

#Calculo de la respuesta a la interseccion de las imágenes de train
kernel_RGB_in = []
kernel_RGB_un = []
kernel_Lab_in = []
kernel_Lab_un = []
kernel_HSV_in = []
kernel_HSV_un = []
chi2_RGB_in = []
chi2_RGB_un = []
chi2_Lab_in = []
chi2_Lab_un = []
chi2_HSV_in = []
chi2_HSV_un = []

for i in tqdm(range(0,len(df_test)),desc='Procesamiento'):
    hrgb = MyColorHist_201715400_201713127(img_test[i],'RGB',False)
    hlab = MyColorHist_201715400_201713127(img_test[i],'Lab',False)
    hhsv = MyColorHist_201715400_201713127(img_test[i],'HSV',False)
    kernel_RGB_in.append(MyIntersectionKernel_201715400_201713127(hrgb,htrain_infected_RGB))
    kernel_RGB_un.append(MyIntersectionKernel_201715400_201713127(hrgb,htrain_uninfected_RGB))
    kernel_Lab_in.append(MyIntersectionKernel_201715400_201713127(hlab,htrain_infected_Lab))
    kernel_Lab_un.append(MyIntersectionKernel_201715400_201713127(hlab,htrain_uninfected_Lab))
    kernel_HSV_in.append(MyIntersectionKernel_201715400_201713127(hhsv,htrain_infected_HSV))
    kernel_HSV_un.append(MyIntersectionKernel_201715400_201713127(hhsv,htrain_uninfected_HSV))
    chi2_RGB_in.append(MyChi2Distance_201715400_201713127(htrain_infected_RGB,hrgb))
    chi2_RGB_un.append(MyChi2Distance_201715400_201713127(htrain_uninfected_RGB,hrgb))
    chi2_Lab_in.append(MyChi2Distance_201715400_201713127(htrain_infected_Lab,hlab))
    chi2_Lab_un.append(MyChi2Distance_201715400_201713127(htrain_uninfected_Lab,hlab))
    chi2_HSV_in.append(MyChi2Distance_201715400_201713127(htrain_infected_HSV,hhsv))
    chi2_HSV_un.append(MyChi2Distance_201715400_201713127(htrain_uninfected_HSV,hhsv))

print('Valores\n')
print('Valores RGB in',kernel_RGB_in,'\n')
print('Valores RGB un',kernel_RGB_un,'\n')
print('Valores Lab in',kernel_Lab_in,'\n')
print('Valores Lab un',kernel_Lab_un,'\n')
print('Valores HSV in',kernel_HSV_in,'\n')
print('Valores HSV un',kernel_HSV_un,'\n')

print('\nValores CHI^2\n')
print('Val RGB in',chi2_RGB_in,'\n')
print('Val RGB un',chi2_RGB_un,'\n')
print('Val Lab in',chi2_Lab_in,'\n')
print('Val Lab un',chi2_Lab_un,'\n')
print('Val HSV in',chi2_HSV_in,'\n')
print('Val HSV un',chi2_HSV_un)

#Funcion para clasificación de acuerdo a los valores de intersección
def clasificacion(valores_in, valores_un, distancia):
    clasificaciones = []

    if  distancia == 'Kernel':
        for i in range(0,len(valores_in)):
            if valores_in[i]>valores_un[i]:
                clasificaciones.append('Parasitized')
            else:
                clasificaciones.append('Uninfected')

    elif distancia == 'CHI2':
        for i in range(0,len(valores_in)):
            if valores_in[i]<valores_un[i]:
                clasificaciones.append('Parasitized')
            else:
                clasificaciones.append('Uninfected')

    return clasificaciones

clasifiacion_kernelRGB = clasificacion(kernel_RGB_in,kernel_RGB_un,'Kernel')
clasifiacion_kernelLab = clasificacion(kernel_Lab_in,kernel_Lab_un,'Kernel')
clasifiacion_kernelHSV = clasificacion(kernel_HSV_in,kernel_HSV_un,'Kernel')
clasifiacion_CHI2RGB = clasificacion(chi2_RGB_in,chi2_RGB_un,'CHI2')
clasifiacion_CHI2Lab = clasificacion(chi2_Lab_in,chi2_Lab_un,'CHI2')
clasifiacion_CHI2HSV = clasificacion(chi2_HSV_in,chi2_HSV_un,'CHI2')

print('  \nClasificaciones\n  ')
print('Kernel_RGB: ', clasifiacion_kernelRGB)
print('Kernel_Lab: ', clasifiacion_kernelLab)
print('Kernel_HSV: ', clasifiacion_kernelHSV)
print('  \nBono\n  ')
print('chi2_RGB: ', clasifiacion_CHI2RGB)
print('chi2_Lab: ', clasifiacion_CHI2Lab)
print('chi2_HSV: ', clasifiacion_CHI2HSV)

#Obtención de las anotaciones
anotaciones = []
import csv
with open('malaria_dataset/annotations.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        anotaciones.append(row[1])
anotaciones.remove(anotaciones[0])

#Obtención de metricas
metricas_kernelRGB = MyConfMatrix_201715400_201713127(anotaciones, clasifiacion_kernelRGB)
metricas_kernelLab = MyConfMatrix_201715400_201713127(anotaciones, clasifiacion_kernelLab)
metricas_kernelHSV = MyConfMatrix_201715400_201713127(anotaciones, clasifiacion_kernelHSV)
metricas_CHI2RGB = MyConfMatrix_201715400_201713127(anotaciones, clasifiacion_CHI2RGB)
metricas_CHI2Lab = MyConfMatrix_201715400_201713127(anotaciones, clasifiacion_CHI2Lab)
metricas_CHI2HSV = MyConfMatrix_201715400_201713127(anotaciones, clasifiacion_CHI2HSV)

#Metricas de histograma de color
print('\nMetricas histigramas de color\n')
print('Kernel_RGB: ', 'P = ',metricas_kernelRGB[3], ', c = ',metricas_kernelRGB[4],', F1 = ',metricas_kernelRGB[5])
print('Kernel_Lab: ', 'P = ',metricas_kernelLab[3], ', c = ',metricas_kernelLab[4],', F1 = ',metricas_kernelLab[5])
print('Kernel_HSV: ', 'P = ',metricas_kernelHSV[3], ', c = ',metricas_kernelHSV[4],', F1 = ',metricas_kernelHSV[5])
print('Chi2_RGB: ', 'P = ',metricas_CHI2RGB[3], ', c = ',metricas_CHI2RGB[4],', F1 = ',metricas_kernelRGB[5])
print('chi2_Lab: ', 'P = ',metricas_CHI2Lab[3], ', c = ',metricas_CHI2Lab[4],', F1 = ',metricas_CHI2Lab[5])
print('chi2_HSV: ', 'P = ',metricas_CHI2HSV[3], ', c = ',metricas_CHI2HSV[4],', F1 = ',metricas_CHI2HSV[5])

#Metricas de template matching: etiquetas ---> Obtenidas con las funciones del lab pasado con pequeñas mejoras.
ppimg_clasificacion = ['Parasitized', 'Parasitized', 'Parasitized', 'Uninfected', 'Parasitized', 'Uninfected', 'Parasitized', 'Uninfected', 'Uninfected', 'Parasitized']
img_clasificacion = ['Parasitized', 'Uninfected', 'Parasitized', 'Parasitized', 'Uninfected', 'Parasitized', 'Parasitized', 'Uninfected', 'Parasitized', 'Parasitized']
metricas_kernelppimg = MyConfMatrix_201715400_201713127(anotaciones,ppimg_clasificacion)
metricas_kernelimg = MyConfMatrix_201715400_201713127(anotaciones,img_clasificacion)
print(' \nMetricas de template matching\n ')
print('Template matching imagen pre-procesada: ', 'P = ',metricas_kernelppimg[3], ', c = ',metricas_kernelppimg[4],', F1 = ',metricas_kernelppimg[5])
print('Template matching: ', 'P = ',metricas_kernelimg[3], ', c = ',metricas_kernelimg[4],', F1 = ',metricas_kernelimg[5])