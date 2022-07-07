#Author: Nicolas Garnica
#Multiclass classification using a feedforward neural network
#Data: MINST-Dataset  -> gray scaled images from handwritten digits (0 to 9). each image is 28 x 28 = 784 pixels.
#Handwritting recognicion problem -> digit classification.

#Import libraries
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import seaborn as sn
import os

#Load the data---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MNIST_df = tf.keras.datasets.mnist
(img_train, lbl_train), (img_test, lbl_test) = MNIST_df.load_data()

#Change image range to [0,1] and flatten images
img_train = img_train/255
img_test = img_test/255

#Data explotration
plt.title('Example of a handwritten digit \nfound on the database: '+ str(lbl_train[0]))
plt.imshow(img_train[0],'gray')
plt.axis('off')
plt.show()

#Create the model-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def MC_ANN_Model():
    ANN_model = tf.keras.Sequential()
    #Flattening of each image on the dataset
    ANN_model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    #Configuration of the number of layers and their activation function
    ANN_model.add(tf.keras.layers.Dense(145,activation='relu'))
    #Set a proportion of neurons that wont be taken in account for the layer's output. (make model faster)
    # ANN_model.add(tf.keras.layers.Dropout(0.1))
    #Set final activation layer wich is the output layer. set activaion softmax for multiple classification. 10 classes to classify.
    ANN_model.add(tf.keras.layers.Dense(10, activation='softmax'))
    ANN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #Train the model
    r = ANN_model.fit(img_train,lbl_train, epochs=20)

    #Save the model and history of the model
    ANN_model.save('Saved_Models/MultipleClassificationANN_SavedModel.h5')
    np.save('Saved_Models/MultipleClassificationANN_SavedTrainHistory.npy',r.history)


#Check if a model is already created
#Define if you want to train the model again or not
if os.path.exists('Saved_Models/MultipleClassificationANN_SavedModel.h5')==False:
    MC_ANN_Model()
elif os.path.exists('Saved_Models/MultipleClassificationANN_SavedModel.h5')==True:
    WantToTrain = str(input('If you want to train the model again, write "True": '))
    if WantToTrain=='True':
        print('Training the multiclass classificator ANN model...')
        MC_ANN_Model()
    else:
        print('\nThe prediction model used is the one saved before.\n')

# #Evaluate the model
ModelHistory =np.load('Saved_Models/MultipleClassificationANN_SavedTrainHistory.npy',allow_pickle='TRUE').item()
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
ANN_new_model = tf.keras.models.load_model('Saved_Models/MultipleClassificationANN_SavedModel.h5')
#Predictions of the test database
predictions = ANN_new_model.predict_classes(img_test)

#Getting index where the prediction was incorrect
indx_bad = []
for i in range(0,len(predictions)):
    if predictions[i]!=lbl_test[i]:
        indx_bad.append(i)

indx_bad1 = int(random.choice(indx_bad))
indx_bad2 = int(random.choice(indx_bad))
indx_bad3 = int(random.choice(indx_bad))

#Exploration of the test database and their corresponding classification by the model and their ground truth label.
fig2, ax2=plt.subplots(2,5)
fig2.suptitle('Predictions and true labels \ngraphical exploration')
ax2[0][0].set_title('Prediction: '+str(predictions[0])+'\nLabel: '+str(lbl_test[0]))
ax2[0][0].axis('off')
ax2[0][0].imshow(img_test[0])
ax2[0][1].set_title('Prediction: '+str(predictions[indx_bad3])+'\nLabel: '+str(lbl_test[indx_bad3]),color='red')
ax2[0][1].axis('off')
ax2[0][1].imshow(img_test[indx_bad3])
ax2[0][2].set_title('Prediction: '+str(predictions[2])+'\nLabel: '+str(lbl_test[2]))
ax2[0][2].axis('off')
ax2[0][2].imshow(img_test[2])
ax2[0][3].set_title('Prediction: '+str(predictions[3])+'\nLabel: '+str(lbl_test[3]))
ax2[0][3].axis('off')
ax2[0][3].imshow(img_test[3])
ax2[0][4].set_title('Prediction: '+str(predictions[indx_bad1])+'\nLabel: '+str(lbl_test[indx_bad1]),color='red')
ax2[0][4].axis('off')
ax2[0][4].imshow(img_test[indx_bad1])
ax2[1][0].set_title('Prediction: '+str(predictions[4])+'\nLabel: '+str(lbl_test[4]))
ax2[1][0].axis('off')
ax2[1][0].imshow(img_test[4])
ax2[1][1].set_title('Prediction: '+str(predictions[5])+'\nLabel: '+str(lbl_test[5]))
ax2[1][1].axis('off')
ax2[1][1].imshow(img_test[5])
ax2[1][2].set_title('Prediction: '+str(predictions[6])+'\nLabel: '+str(lbl_test[6]))
ax2[1][2].axis('off')
ax2[1][2].imshow(img_test[6])
ax2[1][3].set_title('Prediction: '+str(predictions[7])+'\nLabel: '+str(lbl_test[7]))
ax2[1][3].axis('off')
ax2[1][3].imshow(img_test[7])
ax2[1][4].set_title('Prediction: '+str(predictions[indx_bad2])+'\nLabel: '+str(lbl_test[indx_bad2]),color='red')
ax2[1][4].axis('off')
ax2[1][4].imshow(img_test[indx_bad2])

plt.tight_layout()
plt.show()

#Metrics and confussion matrix--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TestAccuracy, TestF1, TestRecall = accuracy_score(lbl_test,predictions), np.round(f1_score(lbl_test,predictions,average='weighted'),4), recall_score(lbl_test,predictions,average='weighted')
cm = confusion_matrix(lbl_test,predictions)
plt.title('Test metrics and confusion matrix \n'+'Accuracy = '+str(TestAccuracy) + '    F1 = '+str(TestF1)+'    Recall = '+str(TestRecall)+'\n')
sn.set(font_scale=1.4)

# create seabvorn heatmap with required labels
x_axis_labels = [0,1,2,3,4,5,6,7,8,9] # labels for x-axis
y_axis_labels = [0,1,2,3,4,5,6,7,8,9] # labels for y-axis
conf = sn.heatmap(cm, annot=True, annot_kws={'size':8}, cmap='Blues',xticklabels=x_axis_labels, yticklabels=y_axis_labels)
conf.set(xlabel='Predicted digit', ylabel='True handwritten digit')
conf.tick_params(left=True, bottom=True)
plt.tight_layout()
plt.show()
