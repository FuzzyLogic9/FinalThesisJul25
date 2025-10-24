import matplotlib.pyplot as plt 
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score,f1_score,cohen_kappa_score
from sklearn.multiclass import OneVsRestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning,ConvergenceWarning

import time

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from itertools import cycle

import numpy as np

import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

#infile="C:\\Users\\1943069\\Scraping_Training_VSC\\Training_Exp1\\PipelineEntryFileFeb22MCNH1Av.csv"
infile = 'H:\\Scraping_Training_VSC\\Training_Exp1\\DatafromWOBtots\\WOB39\\NEWDBSA\\FinalCodes\\PipelineEntryFileFeb22MCNH_OR.csv'

df = pd.read_csv(infile, sep=",", header=None)
X  = df.iloc[:,0:8].values
y = df.iloc[:,33].values


old_y = y
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2, 3, 3, 5, 6, 7])
yb = np.array(y)
#print(yb)
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)


clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                              random_state=0))
#print(y_train)

y_score = clf.fit(X_train, y_train).decision_function(X_test)
#y_score = clf.fit(X_train, y_train)

fpr = dict()
tpr = dict()
roc_auc = dict()

lw=2
for i in range(n_classes):
    #print(fpr[i])
    #print(tpr[i])
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #print(y_test[:, i])
    #print(y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    #print(roc_auc[i]) #FAWZIA - HERE!!!

train_X=X
train_Y=old_y
#print(y)
model = clf
n = 8
cls_rocs = np.zeros((n,2))
  
totalacc = []
totaldiff =[]
totalroc= []
totalpreci= []
totalrec= []
totalf1s= []
totalcks =[]

tot_cls_rocs=[]
start = time.time()
#print("+++")        
#Description
for j in range(1,11):
    acc = []
    diff=[]
    roc = []
    prc =[]
    recl= []
    fones=[]
    cks=[]
    
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    skf.get_n_splits(train_X,train_Y)
    for train_index, test_index in skf.split(train_X,train_Y):
        train_x = train_X[train_index]
        
        train_y = [train_Y[j] for j in train_index]

        test_x = train_X[test_index]
        test_y = [train_Y[j] for j in test_index]

        test_yb = yb[test_index]
   # print(test_yb)

    
    # fit the model on the whole dataset
        Model_Fit = model.fit(train_x,train_y)
    
        y_score = Model_Fit.fit(train_x, train_y).decision_function(test_x)
    #y_score = Model_Fit.fit(train_x, train_y)
        
        The_Prediction =  Model_Fit.predict(test_x)
        Predictions_proba = Model_Fit.predict_proba(test_x)
   
        #accuracy
        pred_acc = Model_Fit.score(test_x, test_y)
    
        preci = precision_score(test_y, The_Prediction,average='weighted')
        recall = recall_score (test_y, The_Prediction,average='weighted')
        fone = f1_score(test_y, The_Prediction,average='weighted')

        cohen_kappascore = cohen_kappa_score(test_y, The_Prediction)

        dif = ((pred_acc)-(test_y)).tolist()

        acc.append(pred_acc)
        diff.append(dif)
        #roc.append(c)
        prc.append(preci)
        recl.append(recall)
        fones.append(fone)
        cks.append(cohen_kappascore)

        roc_auc = dict()
        i=0
        for i in range(n_classes):
        #print(test_yb[:,i])        
        #print(y_score[:,i])
            #print(i)
            fpr[i], tpr[i], _ = roc_curve(test_yb[:,i], y_score[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            #print(roc_auc[i])
            cls_rocs = i,roc_auc[i]
            #cls_rocs.append(i,roc_auc[i])
            #print(i,roc_auc[i]) 
            #print(cls_rocs)
            tot_cls_rocs.append(cls_rocs) 
    
    totalacc.append(acc)
    totaldiff.append(diff)
    totalroc.append(roc)
    totalpreci.append(prc)
    totalrec.append(recl)
    totalf1s.append(fones)
    totalcks.append(cks)
#print(tot_cls_rocs)  
end = time.time()
timetaken = end-start

dfrocs = pd.DataFrame(tot_cls_rocs, columns = ['Class','ROC_AUC'])
print(dfrocs) #print(cls_rocs)
dfbyclass = dfrocs.groupby(["Class"]).ROC_AUC.mean().reset_index()
#print(dfbyclass)

print("Accuracy, Diff, Precision, Recall , F1, Cohen Kappa, Time Taken")
print(np.mean(totalacc),",",np.mean(totaldiff),",",np.mean(totalpreci),",",np.mean(totalrec),",",np.mean(totalf1s),",",np.mean(totalcks), ",",timetaken)
print("ROC Means by Classes")
print(dfbyclass)
