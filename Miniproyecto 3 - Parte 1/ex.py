import numpy as np
from prettytable import PrettyTable

x = np.array([[1,1,1],[1,1,1],[1,1,1]])
y = np.array([[2,1,2],[1,1,1],[1,1,1]])
a = np.array(['Caballo',23,33,4,'Potrico',2,213,34,'Camello',456,53,45])
myTable = PrettyTable(['Num1','Precision','Cobertura','F1'])
print(np.array_equal(x,y))
myTable.add_row([str(a[0]),str(a[1]),str(a[2]),str(a[3])])
myTable.add_row([str(a[4]),str(a[5]),str(a[6]),str(a[7])])
myTable.add_row([str(a[8]),str(a[9]),str(a[10]),str(a[11])])
print(myTable)
