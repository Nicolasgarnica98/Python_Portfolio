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
from main_201715400_201713127 import calculate_descriptors

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
    
    descriptors= np.load(parameters['train_descriptor_name']+'.npy')

    #Creción y/o carga del modelo
    modelo = None
    trained_model = None
    if parameters['Classifier']=='SVM':
        modelo=SVC(kernel=GeneralizedHistogramIntersection())
        trained_model=modelo.fit(descriptors,label_true)
    elif parameters['Classifier'] == 'RF':
        modelo = RandomForestClassifier(n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'], random_state=1111)
        trained_model = modelo.fit(descriptors,label_true)
    elif parameters['Classifier']=='K-Means':
        modelo = KMeans(n_clusters=3,random_state=1111)
        trained_model = modelo.fit(descriptors)

    pickle.dump(trained_model, open(parameters['name_model'], 'wb'))
     # #print(trained_model.labels_)


def validate(parameters, action):
    data_val = os.path.join('data_mp4', 'scene_dataset', 'val', '*.jpg')
    images_val = list(map(io.imread, glob.glob(data_val)))
    # Esta condición solo la tendrán que utilizar para la tercera entrega.
    # TODO Cargar matrices de parameters['val_descriptor_name']
    descriptors=np.load(parameters['val_descriptor_name']+'.npy')
    # TODO Cargar el modelo de parameters['name_model']
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

    if parameters['Classifier']=='K-Means':
        labels = loaded_model.labels_
        print(Val_pred)
        # TODO Obtener las métricas de evaluación
    
        #Correspondencia entre labels y eqtiquetas
        labels2 = []
        inicio = 0
        fin = 8
        for i in range(0,6):
            labels2.append(stats.mode(labels[inicio:fin]))
            inicio = inicio + 8
            fin = fin + 8
    
        True_val = [labels2[0][0][0],labels2[0][0][0],labels2[1][0][0],labels2[1][0][0],labels2[2][0][0],labels2[2][0][0],labels2[3][0][0],labels2[3][0][0],labels2[4][0][0],labels2[4][0][0]
        ,labels2[5][0][0],labels2[5][0][0]]

    
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
    print('eXPERIMENTO: '+str(parameters['k']))
    print(' ')
    print('RESULTADOS OBTENIDOS:')
    print('Presición: '+str(precision))
    print('Recall: '+str(recall))
    print('f1_score: '+str(f_score))
    print(' ')
    print(' ')

entrenar = input('Quiere experimentar? (Insertar si o no): ')


def test(train):
    parameters= {
             'k': 50,
             'name_model': 'final_model_201715400_201713127.pkl', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'Textons-SVM_TrainDesc_K50', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'Textons-SVM_ValDesc_K50',
             'dict_name':'dict_Experimento2_K_50',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'Textons', 
             'Classifier':'SVM', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':8 }
    if train == True:
        data_test = os.path.join('data_mp4', 'scene_dataset', 'test', '*.jpg')
        images_test = list(map(io.imread, glob.glob(data_test)))
        descriptores_test = calculate_descriptors(images_test,parameters=parameters,calculate_dict=False)
        np.save('descriptores_test', descriptores_test)
    else:
        descriptores_test= np.load('descriptores_test.npy')

    modelo = pickle.load(open('final_model_201715400_201713127.pkl', 'rb'))
    val_pred = modelo.predict(descriptores_test)
    True_val =     np.array(([1,1, #building
                              2,2, #forest
                              3,3, #glacier
                              4,4, #mountainst
                              5,5, #sea
                              6,6,])) #street

    conf_mat=confusion_matrix(True_val, val_pred)
    precision=precision_score(True_val, val_pred, average='macro')
    recall=recall_score(True_val, val_pred, average='macro')
    f_score= (2*precision*recall)/(precision+recall)

    print('')
    print('Mejor modelo')
    print('Modelo: Textones + SVM')
    print('Parametros SVM: K = 50,  Kernel = GeneralizedHistogramIntersection')
    print('Convenciones: 1-Building, 2-Forest, 3-Glacier, 4-Mountains, 5-Sea, 6-Street')
    print('Ground truth: 1,1,2,2,3,3,4,4,5,5,6,6')
    print('Predicciones: '+str(val_pred))
    print('Métricas: Precision = ' + str(precision) + ',  Cobertura = '+str(recall)+',  F1 = '+str(f_score))

    return conf_mat, precision, recall, f_score





if entrenar=='si':

    perform_train = False # Cambiar parámetro a False al momento de hacer la entrega
    action = None # Cambiar a None al momento de hacer la entrega 
    calculate_dict=False
    
    parameters1= {
             'k': 'Histograma concatenado + RF',
             'name_model': 'CC_RF', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'CC_RGB_150B_TrainDesc_K3', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'CC_RGB_150B_ValDesc_K3',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters1, perform_train=perform_train, action=action)
    
    parameters2= {
             'k': 'Histograma concatenado + SVM',
             'name_model': 'CC_SVC', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'CC_RGB_150B_TrainDesc_K3', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'CC_RGB_150B_ValDesc_K3',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'SVM', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters2, perform_train=perform_train, action=action)
    
    parameters3= {
             'k': 'Textones + RF',
             'name_model': 'Textones_RF', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'Textons-SVM_TrainDesc_K50', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'Textons-SVM_ValDesc_K50',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters3, perform_train=perform_train, action=action)
    
    parameters4= {
             'k': 'Textones + K-Means',
             'name_model': 'Textonas_Kmeans', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'Textons-SVM_TrainDesc_K50', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'Textons-SVM_ValDesc_K50',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'K-Means', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters4, perform_train=perform_train, action=action)
    
    parameters5= {
             'k': 'HOG_SVM',
             'name_model': 'HOG_SVM', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_L1_5O_descTrain', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_L1_5O_descVal',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'SVM', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters5, perform_train=perform_train, action=action)
    
    parameters6= {
             'k': 'HOG + K-Means',
             'name_model': 'HOG_Kmeans', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_L1_5O_descTrain', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_L1_5O_descVal',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'K-Means', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters6, perform_train=perform_train, action=action)
    
    
    parameters7= {
             'k': 'CC + K-Means',
             'name_model': 'CC_RGB_150B_experimento_K3.pkl', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'CC_RGB_150B_TrainDesc_K3', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'CC_RGB_150B_ValDesc_K3',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'K-Means', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters7, perform_train=perform_train, action=action)
    
    print('//////////////////////////////////////////////////////////// MEJOR MODELO //////////////////////////////////////////////////////////////')
    print('')
    parameters8= {
             'k': 'Textones + SVM',
             'name_model': 'final_model_201715400_201713127.pkl', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'Textons-SVM_TrainDesc_K50', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'Textons-SVM_ValDesc_K50',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'SVM', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters8, perform_train=perform_train, action=action)
    print('////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('')
    
    parameters9= {
             'k': 'HOG + RF',
             'name_model': 'HOG_L1_5O_model', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_L1_5O_descTrain', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_L1_5O_descVal',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L1', 'orientations_HOG':5,
             'n_estimators':100, 'max_depth':4 } # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
    
    main(parameters=parameters9, perform_train=perform_train, action=action)


else:
    print('Analizando imágenes de Test')
    test(False)
