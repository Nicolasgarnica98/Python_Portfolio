import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#leer datos
df = pd.read_excel('lab_7.xlsx')

#Definir variables
Y = df['Y']
X = df['X']
x = sm.add_constant(X)
#Realizar regresion
model = sm.OLS(Y,x).fit()
print(model.summary())
estimaciones = model.predict(x)
plt.plot(X,Y,'.r')
plt.plot(X,estimaciones)
plt.show()
coheficientes = model.params
print(coheficientes)
print('Ecuacion:','Y = ',coheficientes[0],'+',coheficientes[1],'X')
Prediccion = float(input('Insertar valor x a estimar: '))
print('Y = ',coheficientes[0]+(coheficientes[1]*Prediccion))

#%%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

#leer datos
df1 = pd.read_excel('datos_regmultiple.xlsx')

X1 = df1[['B','P','M','S','BP','D','G','MA','SC']]
Y1 = df1['E']
# print(df1.head(5))
X1 = sm.add_constant(X1)
model1 = sm.OLS(Y1,X1).fit()
estimaciones1 = model1.predict(X1)
# print(model1.summary())
# print(estimaciones1)
print('SSE(completo) = ',model1.ssr)
print('R^2(completo) = ',model1.rsquared)


#%%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

X2 = df1[['B','P','M','BP','MA',]]
Y2 = df1['E']
# print(df1.head(5))
X2 = sm.add_constant(X2)
model2 = sm.OLS(Y2,X2).fit()
estimaciones2 = model2.predict(X2)
print(model2.summary())
# print(estimaciones2)
print('SSE(reducido) = ',model2.ssr)
print('R^2(reducido) = ',model2.rsquared)


# %%
