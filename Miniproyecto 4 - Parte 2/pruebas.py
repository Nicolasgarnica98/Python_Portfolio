from main import main
from skimage import color
from data_mp4.functions import JointColorHistogram, CatColorHistogram

#EXPERIMENOS
print('EXPERIMENTOS')
print('')
print('')
print('Histograma conjunto -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('')
print('')

perform_train = True # Cambiar parámetro a False al momento de hacer la entrega
action = 'save' # Cambiar a None al momento de hacer la entrega


#Experimento 1 RGB y histograma conjunto de color 150 BINS
parameters1= {'histogram_function': JointColorHistogram, 
         'space': 'RGB', 'transform_color_function': None, # Esto es solo un ejemplo.
         'bins': 150, 'k': 6,
         'name_model': 'JH_RGB_150B_experimento', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'JH_RGB_150B_TrainDesc', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'JH_RGB_150B_ValDesc'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
 

main(parameters=parameters1, perform_train=perform_train, action=action)



#Experimento 2 HSV y histograma conjunto de color 150 BINS
parameters2= {'histogram_function': JointColorHistogram, 
         'space': 'HSV', 'transform_color_function': color.rgb2hsv, # Esto es solo un ejemplo.
         'bins': 150, 'k': 6,
         'name_model': 'JH_HSV_150B_experimento', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'JH_HSV_150B_TrainDesc', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'JH_HSV_150B_ValDesc'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters2, perform_train=perform_train, action=action)



#Experimento 3 RGB y histograma conjunto de color 10 BINS
parameters3= {'histogram_function': JointColorHistogram, 
         'space': 'RGB', 'transform_color_function': None, # Esto es solo un ejemplo.
         'bins': 10, 'k': 6,
         'name_model': 'JH_RGB_130B_experimento', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'JH_RGB_130B_TrainDesc', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'JH_RGB_130B_ValDesc'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters3, perform_train=perform_train, action=action)



#Experimento 4 HSV y histograma conjunto de color 10 BINS
parameters4= {'histogram_function': JointColorHistogram, 
         'space': 'HSV', 'transform_color_function': color.rgb2hsv, # Esto es solo un ejemplo.
         'bins': 10, 'k': 6,
         'name_model': 'JH_HSV_130B_experimento', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'JH_HSV_130B_TrainDesc', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'JH_HSV_130B_ValDesc'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 
 

main(parameters=parameters4, perform_train=perform_train, action=action)


    
print('')
print('')
print('Histograma concatenado -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('')
print('')
#HSV -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Experimento 5 RGB y histograma concatenado de color 150 BINS
parameters5= {'histogram_function': CatColorHistogram, 
         'space': 'RGB', 'transform_color_function': None, # Esto es solo un ejemplo.
         'bins': 150, 'k': 6,
         'name_model': 'CC_RGB_150B_experimento', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_RGB_150B_TrainDesc', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_RGB_150B_ValDesc'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters5, perform_train=perform_train, action=action)



#Experimento 6 RGB y histograma concatenado de color 150 BINS
parameters6= {'histogram_function': CatColorHistogram, 
         'space': 'HSV', 'transform_color_function': color.rgb2hsv, # Esto es solo un ejemplo.
         'bins': 150, 'k': 6,
         'name_model': 'CC_HSV_150B_experimento', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_HSV_150B_TrainDesc', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_HSV_150B_ValDesc'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters6, perform_train=perform_train, action=action)



#Experimento 7 RGB y histograma concatenado de color 10 BINS
parameters7= {'histogram_function': CatColorHistogram, 
         'space': 'RGB', 'transform_color_function': None, # Esto es solo un ejemplo.
         'bins': 10, 'k': 6,
         'name_model': 'CC_RGB_130B_experimento', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_RGB_130B_TrainDesc', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_RGB_130B_ValDesc'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters7, perform_train=perform_train, action=action)



#Experimento 8 HSV y histograma concatenado de color 10 BINS
parameters8= {'histogram_function': CatColorHistogram, 
         'space': 'HSV', 'transform_color_function': color.rgb2hsv, # Esto es solo un ejemplo.
         'bins': 10, 'k': 6,
         'name_model': 'CC_HSV_130B_experimento', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_HSV_130B_TrainDesc', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_HSV_130B_ValDesc'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters8, perform_train=perform_train, action=action)






print('')
print('')
print('Mejores experimentos - cambio de K -----------------------------------------------------------------------------------------------------------------------------------------------------------')
print('')
print('')


#K=5
parameters9= {'histogram_function': JointColorHistogram, 
         'space': 'RGB', 'transform_color_function': None, # Esto es solo un ejemplo.
         'bins': 150, 'k': 5,
         'name_model': 'CC_RGB_130B_experimento_K5', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_RGB_150B_TrainDesc_K5', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_RGB_150B_ValDesc_K5'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters9, perform_train=perform_train, action=action)

#K=4
parameters10= {'histogram_function': JointColorHistogram, 
         'space': 'RGB', 'transform_color_function': None, # Esto es solo un ejemplo.
         'bins': 150, 'k': 4,
         'name_model': 'CC_RGB_130B_experimento_K4', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_RGB_150B_TrainDesc_K4', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_RGB_150B_ValDesc_K4'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters10, perform_train=perform_train, action=action)



#K=3
parameters11= {'histogram_function': JointColorHistogram, 
         'space': 'RGB', 'transform_color_function': None, # Esto es solo un ejemplo.
         'bins': 150, 'k': 3,
         'name_model': 'CC_RGB_130B_experimento_K3', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_RGB_150B_TrainDesc_K3', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_RGB_150B_ValDesc_K3'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters11, perform_train=perform_train, action=action)



#K=5
parameters9= {'histogram_function': JointColorHistogram, 
         'space': 'HSV', 'transform_color_function': color.rgb2hsv, # Esto es solo un ejemplo.
         'bins': 150, 'k': 5,
         'name_model': 'CC_HSV_150B_experimento_K5', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_HSV_150B_TrainDesc_K5', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_HSV_150B_ValDesc_K5'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters9, perform_train=perform_train, action=action)

#K=4
parameters10= {'histogram_function': JointColorHistogram, 
         'space': 'HSV', 'transform_color_function': color.rgb2hsv, # Esto es solo un ejemplo.
         'bins': 150, 'k': 4,
         'name_model': 'CC_HSV_150B_experimento_K4', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_HSV_150B_TrainDesc_K4', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_HSV_150B_ValDesc_K4'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters10, perform_train=perform_train, action=action)


#K=3
parameters11= {'histogram_function': JointColorHistogram, 
         'space': 'HSV', 'transform_color_function': color.rgb2hsv, # Esto es solo un ejemplo.
         'bins': 150, 'k': 3,
         'name_model': 'CC_HSV_150B_experimento_K3', # No olviden establecer la extensión con la que guardarán sus archivos. 
         'train_descriptor_name': 'CC_HSV_150B_TrainDesc_K3', # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de entrenamiento.
         'val_descriptor_name': 'CC_HSV_150B_ValDesc_K3'} # No olviden asignar un nombre que haga referencia a sus experimentos y que corresponden a las imágenes de validación. 


main(parameters=parameters11, perform_train=perform_train, action=action)