import os
import numpy as np
from skimage import io
from skimage import color
from skimage import img_as_float
from data_mp4.functions import JointColorHistogram, CatColorHistogram
import glob
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score
from scipy import stats

def calculate_descriptors(data, parameters):
    if parameters['space'] != 'RGB':
        data = list(map(parameters['transform_color_function'], data))
    bins = [parameters['bins']]*len(data)
    histograms = list(map(parameters['histogram_function'], data, bins))

    descriptor_matrix = np.array(histograms) 
    
    if parameters['histogram_function']==JointColorHistogram:
        descriptor2=[]
        for i in range(0,descriptor_matrix.shape[0]):
            descriptor2.append(descriptor_matrix[i,:,:,:].flatten())
        descriptor2=np.array(descriptor2) 
        descriptor_matrix=descriptor2
    
    # TODO Verificar tamaño de descriptor_matrix a # imágenes x dimensión del descriptor
    if parameters['histogram_function']==CatColorHistogram:
        if descriptor_matrix.shape[0]==len(data) and descriptor_matrix.shape[1]==parameters['bins']*3:
            print('Tamaño correcto del descriptor (',descriptor_matrix.shape[0],',',descriptor_matrix.shape[1],')')
        else: 
            print('ERROR: Tamaño incorrecto del descriptor (',descriptor_matrix.shape[0],',',descriptor_matrix.shape[1],')')
    else:   
        if descriptor_matrix.shape[0]==len(data) and descriptor_matrix.shape[1]==parameters['bins']**3 :
            print('Tamaño correcto del descriptor (',descriptor_matrix.shape[0],',',descriptor_matrix.shape[1],')')
        else: 
            print('ERROR: Tamaño incorrecto del descriptor (',descriptor_matrix.shape[0],',',descriptor_matrix.shape[1],')')

    return descriptor_matrix
  
def train(parameters, action):

    
    data_train = os.path.join('data_mp4', 'scene_dataset', 'train', '*.jpg')
    images_train = list(map(io.imread, glob.glob(data_train)))
    if action == 'save':
        descriptors = calculate_descriptors(images_train, parameters)  
        # TODO Guardar matriz de descriptores con el nombre 
        np.save(parameters['train_descriptor_name'], descriptors)
            
    else:
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['train_descriptor_name']
        descriptors= np.load(parameters['train_descriptor_name']+'.npy')
        
    # TODO Definir una semilla y utilice la misma para todos los experimentos de la entrega.
    modelo_kmeans=KMeans(n_clusters=parameters['k'],random_state=1111)
    # TODO Inicializar y entrenar el modelo con los descriptores.
    trained_model=modelo_kmeans.fit(descriptors)
    print(trained_model.labels_)
    # TODO Guardar modelo con el nombre del experimento: parameters['name_model']
    pickle.dump(trained_model, open(parameters['name_model'], 'wb'))



def validate(parameters, action):
    data_val = os.path.join('data_mp4', 'scene_dataset', 'val', '*.jpg')
    images_val = list(map(io.imread, glob.glob(data_val)))
    if action == 'load':
        # Esta condición solo la tendrán que utilizar para la tercera entrega.
        # TODO Cargar matrices de parameters['val_descriptor_name']
        descriptors=np.load(parameters['val_descriptor_name']+'.npy')
    else:
        descriptors = calculate_descriptors(images_val, parameters)
        if action == 'save':
            # TODO Guardar matriz de descriptores con el nombre parameters['val_descriptor_name']
            np.save(parameters['val_descriptor_name'], descriptors)
    # TODO Cargar el modelo de parameters['name_model']
    loaded_model = pickle.load(open(parameters['name_model'], 'rb'))
    # TODO Obtener las predicciones para los descriptores de las imágenes de validación
    Val_pred=loaded_model.predict(descriptors)
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
    f_score=f1_score(True_val, Val_pred, average='macro')

    
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
    print('histogram_function: '+str(parameters['histogram_function']).split(' ')[1])
    print('Colo_space: '+parameters['space'])
    print('Bins: '+str(parameters['bins']))
    print('Clusters: '+str(parameters['k']))
    print(' ')
    print('RESULTADOS OBTENIDOS:')
    print('Presición: '+str(precision))
    print('Recall: '+str(recall))
    print('f1_score: '+str(f_score))
    print(' ')
    print(' ')


if __name__ == '__main__':
   
    # TODO Establecer los valores de los parámetros con los que van a experimentar.
    # Nota: Tengan en cuenta que estos parámetros cambiarán según los descriptores
    # y clasificadores a utilizar.
    parameters= {'histogram_function': JointColorHistogram, 
             'space': 'RGB', 'transform_color_function': None, # Esto es solo un ejemplo.
             'bins': 150, 'k': 3,
             'name_model': 'CC_RGB_150B_experimento_K3.pkl', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'CC_RGB_150B_TrainDesc_K3', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'CC_RGB_150B_ValDesc_K3'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 

    perform_train = True # Cambiar parámetro a False al momento de hacer la entrega
    action = 'save' # Cambiar a None al momento de hacer la entrega 
    main(parameters=parameters, perform_train=perform_train, action=action)