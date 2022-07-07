import os
import re
import numpy as np
from skimage import io
from skimage import color
from skimage import img_as_float
import glob
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score
from scipy import stats
from scipy.signal import correlate2d
from scipy.io import loadmat, savemat
from skimage.color import rgb2gray
from scipy.spatial import distance
from sklearn.svm import SVC
from data_mp4.pykernels.regular import GeneralizedHistogramIntersection
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier




def calculateFilterResponse_201715400_291713127(img_gray, filters):
    resp=np.zeros(((img_gray.shape[0]*img_gray.shape[1]),filters.shape[2]))
    for j in range(0,filters.shape[2]):
        corr=correlate2d(img_gray,filters[:,:,j],mode='same')
        corr_2=corr.flatten()
        resp[:,j]=corr_2
        
    return resp


def calculateTextonDictionary_201715400_291713127(images_train, filters, parameters):
    resp_2=np.zeros(((images_train[0].shape[0]*images_train[0].shape[1]*len(images_train)),filters.shape[2]))
    f1=0
    f2=22500
    for i in range(0,len(images_train)):
        img_gray=rgb2gray(images_train[i])
        resp=calculateFilterResponse_201715400_291713127(img_gray, filters)
        for j in range(0,filters.shape[2]):
            resp_2[f1:f2,j]=resp[:,j]
        f1=f1+22500
        f2=f2+22500
            
    modelo_kmeans=KMeans(n_clusters=parameters['k'],random_state=1111)        
    trained_model=modelo_kmeans.fit(resp_2)    
    centroids_arr=modelo_kmeans.cluster_centers_
    centroids={}
    centroids['centroids']=centroids_arr        
    savemat(parameters['dict_name'], centroids)
    
def calculateTextonHistogram_201715400_291713127(img_gray,centroids,parameters):
        
    centroids_dic={}
    for i in range(0,centroids.shape[0]):
        centroids_dic[i]=centroids[i,:]
    filters=loadmat('filterbank.mat')
    filters=filters['filterbank']
    resp=calculateFilterResponse_201715400_291713127(img_gray, filters)
    hist=np.zeros((resp.shape[0]))
    
    
    for i in range (0,hist.shape[0]):
        dist=np.zeros((centroids.shape[0]))
        for j in range(0,dist.shape[0]):
            dist[j]=distance.euclidean(resp[i,:], centroids_dic[j])
        K_min=dist.argmin()
        hist[i]=K_min
    hist,bins=np.histogram(hist,parameters['k'])
    return hist

def calculate_descriptors(data, parameters,calculate_dict):
        
    print('Calculando descriptores...')
    if calculate_dict==True:
        filters=loadmat('filterbank.mat')
        filters=filters['filterbank']
        calculateTextonDictionary_201715400_291713127(data, filters, parameters)
        centroids=loadmat(parameters['dict_name'])['centroids']
    else:
        centroids=loadmat(parameters['dict_name'])['centroids']
    data=list(map(rgb2gray,data))
    descriptor_matrix=np.zeros((len(data),parameters['k']))
    
    
    
    for i in range(0,len(data)):
        descriptor_matrix[i,:]=calculateTextonHistogram_201715400_291713127(data[i],centroids,parameters)
        
    
    
    return descriptor_matrix

def claculate_HOG(img_array, parameters):
    Hog_descriptors = []
    for i in range(0,len(img_array)):
        Hog_descriptors.append(hog(img_array[i],multichannel=True,orientations=parameters['orientations_HOG'],
                               block_norm=parameters['block_norm_HOG']))
    return Hog_descriptors

def train(parameters, action):
    
    data_train = os.path.join('data_mp4', 'scene_dataset', 'train', '*.jpg')
    images_train = list(map(io.imread, glob.glob(data_train)))
    
    #Ground trurth de las imágenes
    label_true=np.array(([1,1,1,1,1,1,1,1, #building
                          2,2,2,2,2,2,2,2, #forest
                          3,3,3,3,3,3,3,3, #glacier
                          4,4,4,4,4,4,4,4, #mountainst
                          5,5,5,5,5,5,5,5, #sea
                          6,6,6,6,6,6,6,6])) #street
    
    #Cálculo de descriptores
    if action == 'save':
        if parameters['descriptor_type'] == 'Textons':
            descriptors = calculate_descriptors(images_train, parameters,calculate_dict)
        elif  parameters['descriptor_type'] == 'HOG':
            descriptors = claculate_HOG(images_train, parameters)
        np.save(parameters['train_descriptor_name'], descriptors)   
    else:
        descriptors= np.load(parameters['train_descriptor_name']+'.npy')

    #Creción y/o carga del modelo
    modelo = None
    trained_model = None
    if parameters['Classifier']=='SVM':
        modelo=SVC(kernel=GeneralizedHistogramIntersection())
        print('Entrenando modelo...')
        trained_model=modelo.fit(descriptors,label_true)
    elif parameters['Classifier'] == 'RF':
        modelo = RandomForestClassifier(n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'])
        trained_model = modelo.fit(descriptors,label_true)

    pickle.dump(trained_model, open(parameters['name_model'], 'wb'))
     # #print(trained_model.labels_)


def validate(parameters, action):
    
    data_val = os.path.join('data_mp4', 'scene_dataset', 'val', '*.jpg')
    images_val = list(map(io.imread, glob.glob(data_val)))
    if action == 'load':
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['val_descriptor_name']
        descriptors=np.load(parameters['val_descriptor_name']+'.npy')
    else:
        if parameters['descriptor_type']=='Textons':
            descriptors = calculate_descriptors(images_val, parameters,calculate_dict)
        elif parameters['descriptor_type']=='HOG':
            descriptors = claculate_HOG(images_val, parameters)
        if action == 'save':
            # TODO Guardar matriz de descriptores con el nombre parameters['val_descriptor_name']
            np.save(parameters['val_descriptor_name'], descriptors)
    # TODO Cargar el modelo de parameters['name_model']
    print('Evaluando...')
    loaded_model = pickle.load(open(parameters['name_model'], 'rb'))
    # TODO Obtener las predicciones para los descriptores de las imágenes de validación
    Val_pred=loaded_model.predict(descriptors)
    
    print(Val_pred)

    
    True_val =     np.array(([1,1, #building
                              2,2, #forest
                              3,3, #glacier
                              4,4, #mountainst
                              5,5, #sea
                              6,6,])) #street
    #Cálculo de métricas
    conf_mat=confusion_matrix(True_val, Val_pred)
    precision=precision_score(True_val, Val_pred, average='macro')
    recall=recall_score(True_val, Val_pred, average='macro')
    f_score= (2*precision*recall)/(precision+recall)

    
    return conf_mat, precision, recall, f_score


def main(parameters, perform_train, action):
    if perform_train:
        train(parameters, action = action)
    conf_mat, precision, recall, f_score = validate(parameters, action = action)
    #TODO Imprimir de manera organizada el resumen del experimento.
    # Incluyan los parámetros que usaron y las métricas de validación.
    
    print(' ')
    print(' ')
    print('NOMBRE DEL EXPERIMENTO: '+parameters['name_model'])
    print(' ')
    print('PARAMETROS UTILIZADOS:')
    #print('histogram_function: '+str(parameters['histogram_function']).split(' ')[1])
    #print('Colo_space: '+parameters['space'])
    #print('Bins: '+str(parameters['bins']))
    print('Clusters: '+str(parameters['k']))
    print(' ')
    print('RESULTADOS OBTENIDOS:')
    print('Presición: '+str(precision))
    print('Recall: '+str(recall))
    print('f1_score: '+str(f_score))
    print(' ')
    print(' ')


if __name__ == '__main__':
    """
    Por: Isabela Hernández y Natalia Valderrama
    Este código establece los parámetros de experimentación. Permite extraer
    los descriptores de las imágenes, entrenar un modelo de clasificación con estos
    y validar su desempeño.
    Nota: Este código fue diseñado para los estudiantes de IBIO-3470 2021-10.
    Rogamos no hacer uso de este código por fuera del curso y de este semestre.
    ----------NO OPEN ACCESS!!!!!!!------------
    """
    # TODO Establecer los valores de los parámetros con los que van a experimentar.
    # Nota: Tengan en cuenta que estos parámetros cambiarán según los descriptores
    # y clasificadores a utilizar.
    parameters= {
             'k': 50,
             'name_model': 'final_model_201715400_201713127.pkl', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'Textons-SVM_TrainDesc_K50', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'Textons-SVM_ValDesc_K50',
             'dict_name':'dict_Experimento2_K_50',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'Textons', 
             'Classifier':'SVM', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':8 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    perform_train = True # Cambiar parámetro a False al momento de hacer la entrega
    action = None # Cambiar a None al momento de hacer la entrega 
    calculate_dict=False
    print('Procesando...')
    main(parameters=parameters, perform_train=perform_train, action=action)


