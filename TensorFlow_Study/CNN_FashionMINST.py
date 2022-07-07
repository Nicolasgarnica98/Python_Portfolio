import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Dense, Input, Flatten, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import seaborn as sn    


#Data load
fahionMINST = tf.keras.datasets.fashion_mnist
(img_Train, lbl_Train), (img_Test, lbl_Test) = fahionMINST.load_data()
print('\nOriginal array shape:')
print('Train image array shape: '+ str(img_Train.shape))
print('Test image array shape: '+ str(img_Test.shape))
num_clases = len(set(lbl_Train))
print('\n# of classes: '+str(len(set(lbl_Train))))

#Data scaling between 1 - 0 for all gray values. Reshaping data for CNN.
img_Train = img_Train/255
img_Test = img_Test/255
img_Train = np.expand_dims(img_Train,-1)
img_Test = np.expand_dims(img_Test,-1)
print('\nNew array shape:')
print('Train image array shape: '+ str(img_Train.shape))
print('Test image array shape: '+ str(img_Test.shape))

#Data exploration
plt.imshow(img_Train[0], cmap='gray')
plt.title('Ground trurh: '+ str(lbl_Train[0]))
plt.axis('off')
plt.show()

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# Model construction
def CNN_Model():
    input = Input(shape=img_Train[0].shape)
    #Convolution layers with stride = 2. No need for pooling
    output = Conv2D(filters=32, kernel_size=(3,3),strides=2, activation='relu')(input)
    output = Conv2D(filters=64, kernel_size=(3,3),strides=2, activation='relu')(output)
    output = Conv2D(filters=128, kernel_size=(3,3),strides=2, activation='relu')(output)
    #Output flattening
    output = Flatten()(output)
    #Dropout for avoiding overfitting (Random neurons will have 0 as weights -> no relevant -> dead neurons)
    output = Dropout(0.2)(output)
    output = Dense(512, activation='relu')(output)
    output = Dropout(0.2)(output)
    output = Dense(len(lbl_Train), activation='softmax')(output)

    model = Model(input,output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    result = model.fit(x=img_Train, y=lbl_Train, epochs=15)

    #Save the model and history of the model
    model.save('Saved_Models/CNNFashionMINST_SavedModel.h5')
    np.save('Saved_Models/CNNFashionMINST_SavedTrainHistory.npy',result.history)


#Check if a model is already created
#Define if you want to train the model again or not
if os.path.exists('Saved_Models/CNNFashionMINST_SavedModel.h5')==False:
    CNN_Model()
elif os.path.exists('Saved_Models/CNNFashionMINST_SavedModel.h5')==True:
    WantToTrain = str(input('If you want to train the model again, write "True": '))
    if WantToTrain=='True':
        print('\nTraining the multiclass classificator ANN model...')
        CNN_Model()
    else:
        print('\nThe prediction model used is the one saved before.\n')


ModelHistory =np.load('Saved_Models/CNNFashionMINST_SavedTrainHistory.npy',allow_pickle='TRUE').item()
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

with tf.device('/CPU:0'):
    #Model load
    new_model = tf.keras.models.load_model('Saved_Models/CNNFashionMINST_SavedModel.h5')
    #Predictions of the test database
    #As sequential is not being used, the attribute predict_classes is not present in de Model class. Instead it returns the probabilities for all the classes and then took the highest of them as the predicted class, using argmax.
    prob_predictions = new_model.predict(img_Test)
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
x_axis_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
y_axis_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
conf = sn.heatmap(cm, annot=True, annot_kws={'size':8}, cmap='Blues',xticklabels=x_axis_labels, yticklabels=y_axis_labels)
conf.set(xlabel='Predicted garment', ylabel='True garment')
conf.tick_params(left=True, bottom=True)
plt.tight_layout()
plt.show()



