
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
import pandas as pd
import os

libro = "lab_7.xlsx"
df = pd.read_excel(os.path.join('Database/'+libro))
print(df)
vueltas= df[["X"]]
print(vueltas)
temperatura = df[["Y"]]
print(temperatura.loc[0])
x = []
y = []

for i in range(len(vueltas)):
    x.append(0)
    y.append(0)

for i in range(len(vueltas)):
    x[i] = float(vueltas.loc[i])
    y[i] = float(temperatura.loc[i])
 
print(x)
print(y)

xprom = np.mean(x)
yprom = np.mean(y)
sx = 0
sxy = 0
plt.plot(x,y,".r")

for i in range(len(x)):
    sx = (sx + (x[i])**2)
    sxy = sxy + x[i]*y[i]

sx = sx/len(x)
sxy = sxy/len(x)

A = np.array([[1,xprom],[xprom,sx]])
b = np.array([[yprom],[sxy]])
print(A)
xsol = lg.solve(A,b)
y1 = xsol[1]*x + xsol[0]
plt.plot(x,y1)
plt.title('Tasa de contagio vs tiempo(Días)')
print ("Pendiente de la recta:", xsol[1])
print("y(x) =",xsol[1],"x","+",xsol[0])
plt.show()