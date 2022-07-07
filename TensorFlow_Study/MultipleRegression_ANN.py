import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Build the data----------------------------------------------------------------------------------------------------
#N observations, and two variables(features)
N = 1000
# Matrix of data of NxD where N=1000 is the number of observations and D=2 the number of features
X_TrainInputMatrix = np.random.random((N,2))*6-3

#Function that we want to interpolate
def Y_func(X_InputMatrix):
    x1 = X_InputMatrix[:,0]
    x2 = X_InputMatrix[:,1]
    y = np.cos(2*x1)+np.sin(3*x2)
    return y
    
Y = Y_func(X_TrainInputMatrix)

#Building the model------------------------------------------------------------------------------------------------
RegANN_model = tf.keras.Sequential()
RegANN_model.add(tf.keras.layers.Dense(128,input_shape=(2,), activation='tanh'))
#Last layer indicates the final shape of the output
RegANN_model.add(tf.keras.layers.Dense(1))

#Fit the model
opt = tf.keras.optimizers.Adam(0.1)
RegANN_model.compile(optimizer=opt,loss='mse')
r = RegANN_model.fit(X_TrainInputMatrix,Y,epochs=100)

#Model evaluation
plt.title('MSE - Loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(r.history['loss'],label='loss')
plt.grid(True)
plt.show()

#Save the model
RegANN_model.save('Saved_Models/MultipleRegressionANN_SavedModel.h5')

#Predictions-------------------------------------------------------------------------------------------------------
#Model load
ANN_new_model = tf.keras.models.load_model('Saved_Models/MultipleRegressionANN_SavedModel.h5')

#Build the matrix of N observations and D features for the test database.
line = np.linspace(-3,3,30)
X1_test, X2_test = np.meshgrid(line,line)
X_TestInputMatrix = np.vstack((X1_test.flatten(),X2_test.flatten())).T

#Predictions
Y_pred = (RegANN_model.predict(X_TestInputMatrix)).flatten()

#Metrics-----------------------------------------------------------------------------------------------------------
#Getting groung truth values
Y_gt = Y_func(X_TestInputMatrix)
#Getting the error mesure: MSE
accuracy = np.round(mean_squared_error(Y_gt,Y_pred),3)

#Graphical inspection of the predicted function--------------------------------------------------------------------
fig, ax1 = plt.subplots(1,3,subplot_kw=dict(projection="3d"))
fig.suptitle('\nTarget function:  '+ r'$y(x_1,x_2)=cos(2x_1)+sen(3x_2)$'+'\n\nTest MSE = '+str(accuracy))
ax1[0].set_title('Training data')
ax1[0].scatter(X_TrainInputMatrix[:,0],X_TrainInputMatrix[:,1],Y)
ax1[1].set_title('Test data')
ax1[1].scatter(X_TestInputMatrix[:,0],X_TestInputMatrix[:,1],Y_gt,antialiased=True)
ax1[2].set_title('Test Prediction')
ax1[2].plot_trisurf(X_TestInputMatrix[:,0],X_TestInputMatrix[:,1],Y_pred,antialiased=True)
fig.tight_layout()
plt.show()

