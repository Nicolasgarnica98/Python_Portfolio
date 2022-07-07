import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix
#Se obtienen las anotaciones y las predicciones encontradas en la carpeta propuesta.
#Se extraen los items necesarios.
mat = scipy.io.loadmat('flower_classifier_results.mat')
gt=mat["groundtruth"]
pred=mat["predictions"]
def MyConfMatrix_Código1_Código2(gt, pred):
    gt_1={}
    pred_1={}
    for i in range(0,gt.shape[1],1):
        gt_1[i]=gt[0,i]
        pred_1[i]=pred[0,i]
    gt=list(gt_1.values())
    pred=list(pred_1.values())
    Unique=np.unique(np.array(gt+pred))
    Unique_num={}
    cont=0
    for i in Unique:
        Unique_num[i]=cont
        cont+=1
    #TRANSFORMACION
    gt_t= np.zeros((len(gt)))
    pred_t= np.zeros((len(pred)))     
    for i in range(0, len(gt),1):
        gt_t[i]=Unique_num[gt[i]] 
        pred_t[i]=Unique_num[pred[i]] 
    conf_matrix= np.zeros((len(Unique_num), len(Unique_num)))           
    for i in range (0,len(gt),1):
        conf_matrix[int(gt_t[i]),int(pred_t[i])]= conf_matrix[int(gt_t[i]),int(pred_t[i])]+1
    ##TRue positives
    TP=np.zeros((len(Unique_num)))
    for i in range(0,len(Unique_num),1):
        TP[i]=conf_matrix[i,i]
    FP = -np.diag(conf_matrix)+conf_matrix.sum(axis=0)##False positives
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)##False begatives
    TN = sum(sum(conf_matrix)) -FP-FN-TP ##True Negative
    prec_class_1= np.round_(TP/(TP+FP), decimals=3)
    rec_class_1= np.round_(TP/(TP+FN), decimals=3)
    mean_prec=np.mean(prec_class_1)
    mean_rec=np.mean(rec_class_1)
    prec_class={}
    rec_class={}
    for i in Unique:
        prec_class[i]=prec_class_1[Unique_num[i]]
        rec_class[i]=rec_class_1[Unique_num[i]]
    
    return conf_matrix,prec_class,rec_class,mean_prec, mean_rec

conf_matrix,prec_class,rec_class,mean_prec, mean_rec=MyConfMatrix_Código1_Código2(gt, pred)






