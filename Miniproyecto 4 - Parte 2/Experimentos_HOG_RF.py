from main_201715400_201713127 import main
from skimage import color
from data_mp4.pykernels.regular import GeneralizedHistogramIntersection

#EXPERIMENOS
print('EXPERIMENTOS')
print('')
print('')
print('HOG-RF L1 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('')
print('')

perform_train = True # Cambiar parámetro a False al momento de hacer la entrega
action = 'save' # Cambiar a None al momento de hacer la entrega


#Experimento 1
parameters1= {'k': 50,
             'name_model': 'HOG_L1_5O_model', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_L1_5O_descTrain', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_L1_5O_descVal',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L1', 
             'orientations_HOG':5, 'n_estimators':100,
             'max_depth':4} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.

main(parameters=parameters1, perform_train=perform_train, action=action)


#Experimento 2
parameters2= {'k': 50,
             'name_model': 'HOG_L2_10O_model', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_L2_10O_descTrain', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_L2_10O_descVal',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L2', 
             'orientations_HOG':10, 'n_estimators':100,
             'max_depth':8} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.

main(parameters=parameters2, perform_train=perform_train, action=action)

#Experimento 3
parameters3= {'k': 50,
             'name_model': 'HOG_L1_5O_RF100-4_model', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_L1_5O_RF100-4_descTrain', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_L1_5O_RF100-4_descVal',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L1', 
             'orientations_HOG':5, 'n_estimators':100,
             'max_depth':4} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.

main(parameters=parameters3, perform_train=perform_train, action=action)

#Experimento 4
parameters4= {'k': 50,
             'name_model': 'HOG_L1_10O_RF200-8_model', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_RF200-8_descTrain', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_RF200-8_descVal',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L1', 
             'orientations_HOG':10, 'n_estimators':200,
             'max_depth':8} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.

main(parameters=parameters4, perform_train=perform_train, action=action)


#Experimento 5
parameters5= {'k': 50,
             'name_model': 'HOG_L2_5O_RF200-8_modell', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_L2_5O_RF200-8_model', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_L2_5O_RF200-8_model',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L2', 
             'orientations_HOG':5, 'n_estimators':200,
             'max_depth':8} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.

main(parameters=parameters5, perform_train=perform_train, action=action)


#Experimento 6
parameters6= {'k': 50,
             'name_model': 'HOG_L2_10O_RF100-4_model', # No olviden establecer la extensión con la que guardarán sus archivos. 
             'train_descriptor_name': 'HOG_L2_10O_RF100-4_model', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
             'val_descriptor_name': 'HOG_L2_10O_RF100-4_model',
             'dict_name':'',
             'kernel':GeneralizedHistogramIntersection, 'descriptor_type':'HOG', 
             'Classifier':'RF', 'block_norm_HOG':'L2', 
             'orientations_HOG':10, 'n_estimators':200,
             'max_depth':4} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación.

main(parameters=parameters6, perform_train=perform_train, action=action)