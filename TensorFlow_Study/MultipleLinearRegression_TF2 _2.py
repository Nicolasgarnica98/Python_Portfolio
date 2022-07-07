import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
print(tf.__version__)
#Multiple linear regression with TensorFlow

#leer datos
df1 = pd.read_excel('datos_regmultiple.xlsx')

X1 = df1[['B','P','M','S','BP','D','G','MA','SC']].values
Y1 = df1['E'].values
print(X1)

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)

#Create the model
LinReg_model = tf.keras.Sequential()
#Input shape
LinReg_model.add(tf.keras.layers.Dense(1,input_shape=(9,)))
#Defining loss function and optimizer
LinReg_model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001),loss='mse')

#Learning rate change accordingly to the epoch
def lr_schedule(epoch, lr):
  # if epoch%200 == 0:
  #   return lr*1.01
  return lr
 
scheduler_ = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

results = LinReg_model.fit(X1,Y1, epochs=400, callbacks=[scheduler_])
test = X1
predicted = LinReg_model.predict(test)

from sklearn.metrics import r2_score

R2 = r2_score(Y1, predicted)
plt.plot(results.history['loss'],label='Train loss')
plt.show()
print('')
print('R2 = ',np.round(R2,4))
print('')
