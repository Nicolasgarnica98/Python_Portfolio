#Author: Nicolas Garnica
#Data analysis using algorithms that implement the math theory of a linear regression.

#Context: We want to build a lineal regression model that can describe the olyimpic performance of different
#atlhetes in the 100m race in plain terrain since 1900. From preveious research, because of the great advancements in medicine
#and science since 1900, the top speed and endurance of a 100m olympic runner has been improving throught the years. In this model
#we would like to create a way to predict the top speed of a 100m olympic runner in the following years. I will asume that the 
#science improvement of the human running capabiltiies trought the years is consistent, and that the relationshp between these improvement
#and the top speed is linear. 

#Libraries import
import matplotlib.pyplot as plt
import numpy.linalg as lg
import numpy as np
##


#year of the olympic's game separated in male and female.
AñosH = np.array([1900,1904,1908,1912,1920,1924,1928,1932,1936,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004])
AñosM = np.array([1928,1932,1936,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004])

#Average time of the 100m race for each olympic game, separated in male and female.
tiempoH = np.array([11.00,11.00,10.80,10.80,10.80,10.60,10.80,10.30,10.30,10.30,10.40,10.50,10.20,10.00,9.95,10.14,10.06,10.25,9.99,9.92,9.96,9.84,9.87,9.85])
tiempoM = np.array([12.20,11.90,11.50,11.90,11.50,11.50,11.00,11.40,11.08,11.07,11.08,11.06,10.97,10.54,10.82,10.94,10.75,10.93])

#Average of the data 
AñosHprom = np.mean(AñosH)
tiempoHprom = np.mean(tiempoH)
AñosMprom = np.mean(AñosM)
tiempoMprom = np.mean(tiempoM)

#Linear regression function. According to theory.
def reglineal (x1,y1):
 sx = 0.0
 sxy = 0.0
 for i in range(len(x1)):
     sx = (sx + (x1[i])**2.0)
     sxy = sxy + x1[i]*y1[i]

 sx = sx/len(x1)
 sxy = sxy/len(x1)
 return sx,sxy

cH = reglineal(AñosH,tiempoH)
MH = np.array([[1.0,AñosHprom],[AñosHprom,reglineal(AñosH,tiempoH)[0]]])
print(MH)
bH = np.array([[tiempoHprom],[reglineal(AñosH,tiempoH)[1]]])
MHu = np.concatenate((MH,bH),axis=1)

cM = reglineal(AñosM,AñosM)
MM = np.array([[1.0,AñosMprom],[AñosMprom,reglineal(AñosM,tiempoM)[0]]])
bM = np.array([[tiempoMprom],[reglineal(AñosM,tiempoM)[1]]])
MMu = np.concatenate((MM,bM),axis=1)

#Function to resolve the matrix
def GaussJordan (AE,x):

 for i in range(0,np.size(AE,0)): #recorrer filas de matriz aumentada
   if AE[i,i] != 0:
     AE[i, :] = (AE[i, :]) / (AE[i, i])
     for j in range(i+1,np.size(AE,0)):
        filaux = AE[j,i]*(AE[i,:])
        AE[j,:] = AE[j,:] - filaux
     if AE[i,i] != 1: #Exepción para la ultima fila de la matriz
       AE[i,:] = AE[i,:]/AE[i,i]
   else:
       print("No hay única solución")

 for i in range(0,x): #Eliminando numeros que estan sobre el pibote
    for j in range(i+1,x):
     AE[i,:] = AE[i,:]-AE[j,:]*AE[i,j]

 return AE

#Getting the coheficients for the linear regression model
rh = GaussJordan(MHu,np.size(MH,0))
Hsol = np.array([rh[0,2],rh[1,2]])
rm = GaussJordan(MMu,np.size(MM,0))
Msol = np.array([rm[0,2],rm[1,2]])

#For comparing male and female results, it is necessary to analyse the same time range.
xt = np.arange(1900,2200,4)
yH = Hsol[1]*xt + Hsol[0]
yM = Msol[1]*xt + Msol[0]

#Getting the pont where both lines will cross each other.
inter = -(Msol[0]-Hsol[0])/(Msol[1]-Hsol[1])
Yinter = Msol[1]*inter + Msol[0]
print("Lines will intersec on:",inter,Yinter)
print("According to this model, by 2156's olympics, female atlhetes will surpass the top speed of the male atlhetes.")


#Graphic
ax1 = plt.subplot(1,2,1),plt.plot(xt,yH,color='gray'),plt.plot(xt,yM,color = 'orange'), plt.title("Linear regression analysis")
plt.xlabel("Olympics")
plt.ylabel("Time for completing 100m race")
plt.plot(inter,Yinter,'ro')
plt.legend(('Male','Female','Intersection'))
plt.annotate('(8.135,2152)',xy=(inter,Yinter),xytext = (2000,Yinter))

#Result comparision with numpy pre-built function
CH = np.polyfit(AñosH,tiempoH,1)
FH = np.poly1d([CH[0],CH[1]])
CM = np.polyfit(AñosM,tiempoM,1)
FM = np.poly1d([CM[0],CM[1]])
ax2 = plt.subplot(1,2,2), plt.plot(xt,FH(xt),color = 'red'), plt.plot(xt,FM(xt),color = 'green')
plt.title("Analysis comparison \nwith Numpy function polyval")
plt.plot(inter,Yinter,'ro')
plt.xlabel("Olympics")
plt.legend(('Male','Female','Intersection'))
plt.annotate('(8.135,2152)',xy=(inter,Yinter),xytext = (2000,Yinter))
plt.tight_layout
plt.show()
