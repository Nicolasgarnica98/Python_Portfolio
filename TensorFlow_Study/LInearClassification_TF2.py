
#Author: Nicolas Garnica
#Diagnosis of breast tissue - classify if it is cancer or not
#Binnary classification problem

#Import tensorflow
import tensorflow as tf
#Tensorflow version
print(tf.__version__)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import save_model, load_model
#Load the data
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
#Data comes in a dictionary
data_df = load_breast_cancer()
print(data_df.keys())
#Get features: 30 metrics and meditions of the breast tumor cells.
features = data_df['feature_names']
#Get the data. each data per patient has 30 feature messurements.
data_array = data_df['data']
#Labels of the data
labels = data_df['target']

#Dividing the data into train and test.s
d_train, d_test, lbl_train, lbl_test =  train_test_split(data_array,labels,test_size=0.2)

#Data scaling: Output is a linear combination of the input, we do not want inputs with a very very long range e.q.
# one million and other inputs of samall range e.g. 0.0001. It is necessary to normalize the data. Because the weights
# will be too sensitive when the input has a large range, or not enough sensitive because of very small range.
#Normalization:
#            X - mu
#       Z = -------- ->  mu = Mean, sigma = Standard Deviation
#            sigma
#Normalization function in sklearn: StandardScaler

Normalizer_scaler = StandardScaler()
d_train = Normalizer_scaler.fit_transform(d_train)
d_test = Normalizer_scaler.transform(d_test)

#Building the model
binnary_model = tf.keras.models.Sequential()
binnary_model.add(tf.keras.layers.Dense(1,input_shape=(d_train.shape[1],), activation='sigmoid'))

binnary_model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training and get results from both training and test
Results = binnary_model.fit(d_train,lbl_train, validation_data=(d_test,lbl_test),epochs=200)

#Make a prediction
predictions = binnary_model.predict_classes(d_test[[0]])
predictions = predictions.flatten()
lbl_predictions = []
for i in range(0,len(predictions)):
    if predictions[i] == 1:
        lbl_predictions.append('Cancer')
    else:
        lbl_predictions.append('Negative')

print(lbl_predictions[0])

#Evaluate the model
#print('Train score: ', binnary_model.evaluate(d_train,lbl_train))
#print('Train score: ', binnary_model.evaluate(d_test,lbl_test))

#Loss and accuracy plots per epoch
Fig1, ax1 = plt.subplots(1,2)
ax1[0].set_title('Loss plot')
ax1[0].plot(Results.history['loss'],label='Train loss')
ax1[0].plot(Results.history['val_loss'], label='Test loss')
ax1[0].set_xlabel('Epoch')
ax1[0].set_ylabel('Loss')
ax1[0].legend()
ax1[1].set_title('Accuracy plot')
ax1[1].plot(Results.history['accuracy'],label='Train accuracy')
ax1[1].plot(Results.history['val_accuracy'], label='Test accuracy')
ax1[1].set_xlabel('Epoch')
ax1[1].set_ylabel('Accuracy')
ax1[1].legend()
Fig1.tight_layout()
plt.show()
