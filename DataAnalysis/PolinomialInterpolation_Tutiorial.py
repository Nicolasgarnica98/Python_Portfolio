## x
import numpy as np
import matplotlib.pyplot as plt

##

#1. Para un sistema matricial de la forma Ax=b, donde A es una matriz de coeficientes constantes de NxN,
# x un vector de incógnitas de Nx1 y b un vector de contantes de Nx1, realice una función en Python que
# devuelva el vector de solución x usando el método de Gauss. Pruebe la función con una matriz A aleatoria (rand)
# de 3x3 y un vector b aleatorio de 3x1. Compare la solución encontrada con las funciones de la librería numpy del
# paquete linalg.

# #Para matriz A y vector b aleatorios
# A = np.random.rand(3,3)
# b = np.random.rand(3,1)

#Matríz A y vector b para unos valores dados
A = np.array([[1,1,1],[3,2,1],[4,3,1]])
b = np.array([[60],[95],[125]])


#Se define la función Gauss que implementa el método de solución de Gauss para reducir sistemas de ecuaciones diferenciales.

def Gauss(A,b,N):
    Au=np.concatenate((A,b),1)                  # Concatenación de la matriz A de coeficientes y el vector solución b.
    for i in range(0, np.size(Au, 0)):          # Ciclo que recorre cada fila
        for j in range(i+1, np.size(Au, 0)):    # Ciclo que recorre cada columna
            filaux=(1.0/Au[i,i])*Au[i,:]        # Cada pivote es dividido por él mismo
            filaux=(-1.0*Au[j,i])*filaux        # Se deja 0 debajo de los pivotes
            Au[j,:]=Au[j,:]+filaux              # Se reemplaza por el valor dado

    print(Au)                                   # Se imprime la matriz aumentada que arroja el método de Gauss

    x=np.zeros((N,1))                           # Se crea una matriz de ceros que será posteriormente actualizada
    for i in range(np.size(Au, 0) - 1, -1, -1): # Ciclo que permite obtener la solución de cada variable
        sumaux=0
        for j in range(i+1, np.size(A, 0)):     # Se recorren las columnas
            sumaux=sumaux+Au[i, j]*x[j]

        x[i]=(1.0/Au[i, i])*(Au[i, np.size(Au, 1) - 1]-sumaux)  # Se obtiene el vector 'x' que tiene la solución de las incógnitas
    print("Solución por Gauss: \n", x)
    return [x]

x1=np.linalg.solve(A,b)                         # Paquete linalg de Numpy para solucionar el sistema de ecuaciones
print("Solución de Numpy: \n",x1)


############################################
# 2. Para un sistema matricial de la forma Ax=b, donde A es una matriz de coeficientes constantes de NxN,
# x un vector de incógnitas de Nx1 y b un vector contantes de Nx1, realice una función en Python que devuelva
# el vector de solución x usando el método de Gauss-Jordan. La función debe devolver igualmente la matriz inversa A-1.
# Pruebe la función con una matriz A aleatoria (rand) de 3x3 y un vector b aleatorio de 3x1. Compare la solución
# encontrada con las funciones de la librería numpy del paquete linalg.

def Gauss_Jordan(A,b):
    Au=np.concatenate((A,b),1)                  # Concatenación de la matriz A de coeficientes y el vector solución b.
    for i in range(0, np.size(Au, 0)):          # Ciclo que recorre cada fila
        Au[i, :] = (1.0/Au[i,i])*Au[i, :]       # Cada pivote es dividido por él mismo
        for j in range(0, np.size(Au, 0)):      # Ciclo que recorre cada columna
            if i == j:                          # Condición que evalúa los pivotes. Si es un pivote, salta a la siguiente fila
                continue
            filaux=(-1.0*Au[j,i])*Au[i,:]       # Se deja 0 debajo de los pivotes
            Au[j,:]=Au[j,:]+filaux              # Se reemplaza por el valor dado

    x2 = Au[:, np.size(Au, 1)-1]                # Se obtiene el vector x2 que contiene la solución de las incógnitas.
    print(Au)

    print("Solución por Gauss-Jordan: \n", x2)
    return [x2]

x1=np.linalg.solve(A,b)                         # Paquete linalg de Numpy para solucionar el sistema de ecuaciones
print("Solución de Numpy: \n",x1)

##
#5. Un polinomio de orden 4 está dado por la ecuación P(x) = c4 x^4 + c3 x^3 + c2 x^2 + c1 x + c0 , donde
# c4, c3, c2, c1 y c0 corresponden a coeficientes constantes. Encuentre la ecuación P(x) del polinomio de orden
# 4 que pasa por los puntos (-2.68, 0), (-3.25, 1.15), (-4.45, -1.56), (-6.25, -2.84) y (-8.15, 0.23).
# Utilice las funciones desarrolladas en los puntos 1 y 2 para hallar los coeficientes requeridos. Compare
# la solución encontrada con las funciones de la librería numpy del paquete linalg. Realice una gráfica del
# polinomio encontrado para x=−8.15:0.1:−2.68 .

'''
Puntos:
x0 = (1,0)
x1 = (3,1.0986)
x2 = (4,1.3863)
x3 = (6,1.7918)
x4 = (7,1.9459)
'''

# Se ingresa la matríz de coeficientes A y el vector solución b
# Vectores columna (#), son la componente x de los puntos x0 a x4
# y los otros numeros sin parentesis son las potencias de las x de mayor a menor por fila.
A = np.array([[(1)**4, (1)**3, (1)**2, (1), 1],
              [(3)**4, (3)**3, (3)**2, (3), 1],
              [(4)**4, (4)**3, (4)**2, (4), 1],
              [(6)**4, (6)**3, (6)**2, (6), 1],
              [(7)**4, (7)**3, (7)**2, (7), 1]])

b = np.array([[0],[1.0986],[1.3863],[1.7918],[1.9459]])

# Solución por el método de Gauss
#X = Gauss(A,b,5)

# Solución por el método de Gauss - Jordan
X = Gauss_Jordan(A,b)

#Se define el polinomio y los valores de la variable independiente 'x'
x = np.linspace(0.5,10,1000)
Px = X[0][0] * (x)**4 + X[0][1] * (x)**3 + X[0][2] * (x)**2 + X[0][3] * (x) + X[0][4]

# Gráfica del polinomio
plt.plot(x,Px, color='y',label='Ajuste de ln(x) con Polinomio de segundo grado')
plt.plot(x,np.log(x), color='c',label='Logaritmo natural ln(x)')
plt.legend()
plt.grid(True)
plt.show()

##
# Hallar el ln(2) con n+1 puntos

'''
Puntos:
x0 = (1,0)
x1 = (4,1.3863)
x2 = (6,1.7918)
'''

# Se ingresa la matríz de coeficientes A y el vector solución b
A = np.array([[(1)**2, (1)**1, 1],
              [(4)**2, (4)**1, 1],
              [(6)**2, (6)**1, 1]])

b = np.array([[0],[1.3863],[1.7918]])

# Solución por el método de Gauss - Jordan
X = Gauss_Jordan(A,b)

#Se define el polinomio y los valores de la variable independiente 'x'
x = np.linspace(0.5,10,1000)
Px = X[0][0] * (x)**2 + X[0][1] * (x) + X[0][2]

# Gráfica del polinomio
plt.plot(x,Px, color='y',label='Ajuste de ln(x) con Polinomio de segundo grado')
plt.plot(x,np.log(x), color='c',label='Logaritmo natural ln(x)')
plt.legend()
plt.grid(True)
plt.show()
## Lagrange
# Interpolacion de Lagrange
# Polinomio en forma simbólica
# Ejemplo: dados los 4 puntos en la tabla se requiere generar un polinomio de grado 3 de la forma:
# p(x)=a0x3+a1x2+a2x1+a3
#xi= 0 0.2 0.3 0.4
#fi=1 1.6 1.7 2.0
import sympy as sym

# datos de prueba
xi = np.array([0, 0.2, 0.3, 0.4])
fi = np.array([1, 1.6, 1.7, 2.0])

# PROCEDIMIENTO
n = len(xi)
x = sym.Symbol('X')
# Polinomio
polinomio = 0
for i in range(0,n,1):
    # Termino de Lagrange
    termino = 1
    for j  in range(0,n,1):
        if (j!=i):
            termino = termino*(x-xi[j])/(xi[i]-xi[j])
    polinomio = polinomio + termino*fi[i]
# Expande el polinomio
px = polinomio.expand()
# para evaluacion numérica
pxn = sym.lambdify(x,polinomio)

# Puntos para la gráfica
a = np.min(xi)
b = np.max(xi)
muestras = 101
xi_p = np.linspace(a,b,muestras)
fi_p = pxn(xi_p)

# Salida
print('Polinomio de Lagrange, expresiones')
print(polinomio)
print()
print('Polinomio de Lagrange: ')
print(px)

# Gráfica
plt.title('Interpolación Lagrange')
plt.plot(xi,fi,'o', label = 'Puntos')
plt.plot(xi_p,fi_p, label = 'Polinomio')
plt.legend()
plt.show()

