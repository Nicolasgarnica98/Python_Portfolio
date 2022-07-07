import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
import seaborn as sn

#Data load
fahionMINST = tf.keras.datasets.cifar10
(img_Train, lbl_Train), (img_Test, lbl_Test) = fahionMINST.load_data()

print('\nOriginal array shape:')
print('Train image array shape: '+ str(img_Train.shape))
print('Test image array shape: '+ str(img_Test.shape))
print('Ground truth Train: '+ str(lbl_Train.shape))
print('Ground truth Test: '+ str(lbl_Test.shape))


#Data scaling between 1 - 0. Reshaping data for CNN.
img_Train = img_Train/255
img_Test = img_Test/255
lbl_Train = lbl_Train.flatten()
lbl_Test = lbl_Test.flatten()
print('\nNew label array shape:')
print('Ground truth Train: '+ str(lbl_Train.shape))
print('Ground truth Test: '+ str(lbl_Test.shape))

# #Data exploration
fig1, ax1=plt.subplots(1,4)
fig1.suptitle('\nData exploration')
ax1[0].imshow(img_Train[1])
ax1[0].axis('off')
ax1[1].imshow(img_Train[2])
ax1[1].axis('off')
ax1[2].imshow(img_Train[3])
ax1[2].axis('off')
ax1[3].imshow(img_Train[4])
ax1[3].axis('off')
plt.tight_layout()
plt.show()

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Model construction
def CNN_Model():
    

    i = Input(shape=img_Train[0].shape)
    #Convolution layers with stride = 2. No need for pooling
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(i)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    #Output flattening
    x = Flatten()(x)
    #Dropout for avoiding overfitting (Random neurons will have 0 as weights -> no relevant -> dead neurons)
    x = Dropout(0.2)(x)
    x = Dense(1700, activation='relu')(x)
    x = Dropout(0.2)(x)
    #Softmax activation for the last layer for multiclass classification.
    x = Dense(len(lbl_Train), activation='softmax')(x)

    batch_size = 32
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_generator = data_generator.flow(img_Train,lbl_Train,batch_size)
    steps_per_epoch = img_Train.shape[0]//batch_size
    
    model = Model(i,x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # result = model.fit(x=img_Train, y=lbl_Train, epochs=45)
    result = model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch, epochs=50)

    #Save the model and history of the model
    model.save('Saved_Models/CNN_CIFAR10_(Improved)SavedModel.h5')
    np.save('Saved_Models/CNN_CIFAR10_(Improved)SavedTrainHistory.npy',result.history)

    #Check if a model is already created
#Define if you want to train the model again or not
if os.path.exists('Saved_Models/CNN_CIFAR10_(Improved)SavedModel.h5')==False:
    CNN_Model()
elif os.path.exists('Saved_Models/CNN_CIFAR10_(Improved)SavedModel.h5')==True:
    WantToTrain = str(input('\nIf you want to train the model again, write "True": '))
    if WantToTrain=='True':
        print('\nTraining the multiclass classificator ANN model...')
        CNN_Model()
    else:
        print('\nThe prediction model used is the one saved before.\n')

ModelHistory =np.load('Saved_Models/CNN_CIFAR10_(Improved)SavedTrainHistory.npy',allow_pickle='TRUE').item()
fig1, ax1=plt.subplots(1,2)
fig1.suptitle('Model evaluation')
ax1[0].set_title('Accuracy per epoch')
ax1[0].plot(ModelHistory['accuracy'],label='Accuracy')
ax1[0].set_xlabel('Epoch')
ax1[0].set_ylabel('Accuracy')
ax1[0].legend()
ax1[0].grid(True)
ax1[1].plot(ModelHistory['loss'],label='Loss',color='orange')
ax1[1].set_title('Loss per epoch')
ax1[1].set_xlabel('Epoch')
ax1[1].set_ylabel('Loss')
ax1[1].legend()
ax1[1].grid(True)
fig1.tight_layout()
plt.show()



#Test model and predictions------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Model load
with tf.device('/CPU:0'):
  new_model = tf.keras.models.load_model('Saved_Models/CNN_CIFAR10_(Improved)SavedModel.h5')

  #Predictions of the test database
  #As sequential is not being used, the attribute predict_classes is not present in de Model class. Instead it returns the probabilities for all the classes and then took the highest of them as the predicted class, using argmax.
  print("Testing model")
  prob_predictions = new_model.predict(img_Test)
  print("Testing model2")
  predictions = prob_predictions.argmax(axis=1)
  # for i in range(0,img_Test.shape[0]):
  #     predictions.append(int(np.argmax(prob_predictions[i])))

#Calculating the accuracy, recall and f1 score.
TestAccuracy, TestF1, TestRecall = accuracy_score(lbl_Test,predictions), np.round(f1_score(lbl_Test,predictions,average='weighted'),4), recall_score(lbl_Test,predictions,average='weighted')
cm = confusion_matrix(lbl_Test,predictions)
plt.title('Test metrics and confusion matrix \n'+'Accuracy = '+str(TestAccuracy) + '    F1 = '+str(TestF1)+'    Recall = '+str(TestRecall)+'\n')
sn.set(font_scale=1.4)

# create seaborn heatmap with required labels
# x_axis_labels = np.arange(0,9,dtype=int) # labels for x-axis
# y_axis_labels = np.arange(0,9,dtype=int) # labels for y-axis
x_axis_labels = ['Airplane', 'Automobile', 'Bird', 'Car', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
y_axis_labels = ['Airplane', 'Automobile', 'Bird', 'Car', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
conf = sn.heatmap(cm, annot=True, annot_kws={'size':8}, cmap='Blues',xticklabels=x_axis_labels, yticklabels=y_axis_labels)
conf.set(xlabel='Predicted garment', ylabel='True garment')
conf.tick_params(left=True, bottom=True)
plt.tight_layout()
plt.show()