import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
import pandas as pd
from sklearn.linear_model import LinearRegression as LG

libro = "lab_7.xlsx"
df = pd.read_excel(libro)
print(df)
vueltas= df[["Dia"]]
print(vueltas)
temperatura = df[["Tasa"]]
print(temperatura.loc[0])

#regresion con libreria sklearn
reg = LG().fit(vueltas, temperatura)
#valores predichos para Y, realizados por la regresion
predccion_lineal = reg.predict(vueltas)
#Coheficiente R^2
print('R^2 = ',reg.score(vueltas, temperatura,))

print('Pendiente recta:', reg.coef_)
print('Intercepto en y:', reg.intercept_)
print('Ecuacion de la recta: ', 'Tasa(Dias) =', reg.coef_[0][0],'*','Dias',' + ',reg.intercept_[0])

plt.plot(vueltas,temperatura, '.r'), plt.plot(vueltas, predccion_lineal)
plt.show()

