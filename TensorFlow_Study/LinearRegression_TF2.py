from sched import scheduler
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from sklearn.preprocessing import StandardScaler

#Get data with !wget ---> Get data from the web.
URL = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv'
urlretrieve(URL,'LinearReg_Data.csv')
df = pd.read_csv('LinearReg_Data.csv', usecols=[0,1], names=['Year', 'ex_Growth'])

#Getting the values of the data
X = df.Year
X = X.values
#Reshaping the X data. each array inside the bigger array, is the data entry for TensorFlow (Makes a 2D array of size N x D, where D = 1 in this case beacuse
# we are testing only one variable/feature.)
X = X.reshape(-1,1)
#Y data stays the same! as a 1D array.
Y = df.ex_Growth
Y = Y.values

#As data exhibits exponential behavior, we convert Y into ln(Y) for adress a linear relation
Y = np.log(Y)
X = X - np.mean(X)

#Create the model
LinReg_model = tf.keras.Sequential()
LinReg_model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
LinReg_model.compile(optimizer=tf.keras.optimizers.SGD(0.001,0.9),loss='mse')

#Learning rate change after 50 epoch
def schedule(epoch, lr):
  if epoch >= 50:
    return 0.0001
  return 0.001
 
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

results = LinReg_model.fit(X,Y, epochs=200, callbacks=[scheduler])

plt.plot(results.history['loss'], label='loss')
plt.show()
