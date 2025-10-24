from dask.dataframe.methods import values
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning,ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from numpy import loadtxt

from sklearnex import patch_sklearn 
patch_sklearn()

from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
#from sklearn.naive_bayes import GaussianNB, BernoulliNB
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection._validation import cross_val_score 

from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, f1_score,roc_auc_score



import pandas as pd
import numpy as np
import sys
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

#import sklearn
#print(sklearn.__version__)
#model1 = BernoulliNB()

#MLT1= "BNB"
model1 = RandomForestClassifier() ##

MLT1="RFC"

def RunClass(model_no,train_x,test_x,train_y,test_y):
   
  
    model = model_no
    # fit the model on the whole dataset
    Model_Fit = model.fit(train_x,train_y)
    
    # make a single prediction
    i = 0
    a = 0
 
    The_Prediction =  Model_Fit.predict(test_x)
    Predictions_proba = Model_Fit.predict_proba(test_x)

    test_y_arr= np.array(test_y)
    #pred_probs_arr= np.array(Predictions_proba)

   
    #accuracy
    pred_acc = Model_Fit.score(test_x, test_y)
    
    roc_auc_scores=roc_auc_score(test_y, Predictions_proba, multi_class="ovr", average="weighted")
    preci = precision_score(test_y, The_Prediction,average='weighted')
    recall = recall_score (test_y, The_Prediction,average='weighted')
    fone = f1_score(test_y, The_Prediction,average='weighted')

    cohen_kappascore = cohen_kappa_score(test_y, The_Prediction, weights='quadratic')


    #diff = ((The_Prediction)-(test_y)).tolist()
    diff = ((pred_acc)-(test_y)).tolist()
    return(pred_acc,((diff.count(0)) / len(diff)),(roc_auc_scores),preci, recall, fone, cohen_kappascore)



def Fold_10Looper(model_no,train_X,train_Y):
    
    # starting time
    start = time.time()
    
    totalacc = []
    totaldiff =[]
    totalroc= []
    totalpreci= []
    totalrec= []
    totalf1s= []
    totalcks =[]

    

    for i in range(1,11):
        
        acc = []
        diff=[]
        roc = []
        prc =[]
        recl= []
        fones=[]
        cks=[]

    # 10 - Fold!! times to do the StratifiedKFold which is a variation of KFold. 
    # First, StratifiedKFold shuffles your data, after that splits the data into n_splits
    # parts and Done.
    # Now, it will use each part as a test set.
    # Note that it only and always shuffles data one time before splitting.
    # the difference here is that StratifiedKFold just shuffles and splits once, 
    # therefore the test sets do not overlap, while StratifiedShuffleSplit shuffles each time 
    # before splitting, and it splits n_splits times, the test sets can overlap. 

        skf = StratifiedKFold(n_splits=10,shuffle=True)
        skf.get_n_splits(train_X,train_Y)
        
        #Description
        for train_index, test_index in skf.split(train_X,train_Y):
            train_x = train_X[train_index];
        
            train_y = [train_Y[j] for j in train_index];

            test_x = train_X[test_index];
            test_y = [train_Y[j] for j in test_index];
       
            a,b,c,d,e,f,g = RunClass(model_no, train_x,test_x,train_y,test_y)
           
            acc.append(a)
            diff.append(b)
            roc.append(c)
            prc.append(d)
            recl.append(e)
            fones.append(f)
            cks.append(g)

            # Nested CV with parameter optimization
            #nested_score = cross_val_score(CLF, X=train_x, y=train_y, cv=outer_cv, scoring=make_scorer(classification_report_with_accuracy_score))


 # return accuracy score
    
        totalacc.append(acc)
        totaldiff.append(diff)
        totalroc.append(roc)
        totalpreci.append(prc)
        totalrec.append(recl)
        totalf1s.append(fones)
        totalcks.append(cks)
     # end time
    end = time.time()
    #print(start)
    #print(end)
    timetaken = end-start
  
    print(np.mean(totalacc),",",np.mean(totaldiff),",",np.mean(totalroc),",",np.mean(totalpreci),",",np.mean(totalrec),",",np.mean(totalf1s),",",np.mean(totalcks), ",",timetaken)






items = []

#infile = 'H:\\Scraping_Training_VSC\\Training_Exp1\\DatafromWOBtots\\WOB39\\NEWDBSA\\FinalCodes\\PipelineEntryFileFeb22MCNH_OR.csv'
infile="C:\\Fuzzy\\PipelineEntryFileFeb22MCNH15.csv"
#infile = 'H:\\Scraping_Training_VSC\\Training_Exp1\\DatafromWOBtots\\WOB39\\NEWDBSA\\Removedheadings\\PipelineEntryFile_Headingtest.csv'

print("Hello World!")


df = pd.read_csv(infile, sep=",", header=None)
# [0] Aggression Stanford Core NLP Mean(0-4) Sentence Score || [1]Aggression Vader Negative || [2]Aggression Vader Neutral ||[3]Aggression Vader Positive ||
# [4] Aggression Vader Compound || [5]Aggression TextBlob Polarity || [6]Aggression Polyglot Polarity	|| [7]Aggression Pattern Polarity ||
# [8] Attack Stanford Core NLP Mean(0-4) Sentence Score ||	[9]Attack Vader Negative ||	[10]Attack Vader Neutral ||	[11]Attack Vader Positive !!
# [12] Attack Vader Compound !!	[13]Attack TextBlob Polarity || [14] Attack Polyglot Polarity	|| [15] Attack Pattern Polarity ||
# [16] Toxic Stanford Core NLP Mean(0-4) Sentence Score ||	[17]Toxic Vader Negative ||	[18]Toxic Vader Neutral ||	[19]Toxic Vader Positive ||	
# [20] Toxic Vader Compound	|| [21] Toxic TextBlob Polarity	|| [22]Toxic Polyglot Polarity	|| [23]Toxic Pattern Polarity ||
# [24] Humannotated Aggression Class  Average || [25] Humannotated Aggression Class Average Wholed || [26] Humannotated Aggression Class ||
# [27] Humannotated Attack Class Average || [28] Humannotated Attack Class Average Wholed || [29] Humannotated Attack Class ||
# [30] Humannotated Toxic Class Average || [31]	Humannotated Toxic Class Average Wholed [32] Humannotated Toxic Class || [33] Multi-Class

#outfile = 'H:\\Scraping_Training_VSC\\Training_Exp1\\DatafromWOBtots\\WOB39\\NEWDBSA\\FinalCodes\\OutputforloopallwoMCMLP_Testhead.csv'
#outfile = 'H:\\Scraping_Training_VSC\\Training_Exp1\\DatafromWOBtots\\WOB39\\NEWDBSA\\FinalCodes\\OutputforloopallwoMC1_OR.csv'
outfile = "C:\\Fuzzy\\Multi_RFC_Plain_Out_Final.csv"
#sys.stdout = open(outfile, "a")


#print("1. Train_X_with_only_Aggression ") 
train_X1  = df.iloc[:,0:8].values
    #train_X = train_X_with_only_Aggression
DST1="Agg"
#print("2. Train_X_with_only_Attacks") 
train_X2  = df.iloc[:,8:16].values
    #train_X = train_X_with_only_Attacks
DST2="Att"
#print("3. Train_X_with_only_Toxicity")  
train_X3  = df.iloc[:,16:24].values
    #train_X = train_X_with_only_Toxicity
DST3="Tox"
#print("4. Train_X_with_only_Aggression_and_Attacks") 
train_X4  = df.iloc[:,0:16].values
    #train_X = train_X_with_only_Aggression_and_Attacks
DST4="AggAtt"
#print("5. Train_X_with_only_Aggression_and_Toxicity") 
train_X5  = df.iloc[:,[0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23]].values
    #train_X = train_X_with_only_Aggression_and_Toxicity 
DST5="AggTox"
#print("6. Train_X_with_only_Attacks_and_Toxicity") 
train_X6  = df.iloc[:,8:24].values
    #train_X = train_X_with_only_Attacks_and_Toxicity 
DST6="AttTox"
#print("7. Train_X_with_ALL_3") 
train_X7  = df.iloc[:,0:24].values
    #train_X = train_X_with_ALL_3
DST7="ALL"
#print("1. Train_Y_with_only_Aggression ") 
train_Y1 = df.iloc[:,33].values
CLT1 ="MutiClass"



#Do all your dataset, class model choice and measurements with ALL option

count =1
#sys.stdout = open(outfile, "a")
#print(train_X1[0])
#print(train_X2[0])
#print(train_X3[0])
#print(train_X4[0])
#print(train_X5[0])
#print(train_X6[0])
#print(train_X7[0])
#print(train_Y1[0])
#print(train_Y2[0])
#print(train_Y3[0])
#print(train_Y4[0])
print("Count, DataSet, Class, Model, Accuracy, Diff, ROC, Precision, Recall , F1, Cohen Kappa, Time Taken")
for i in range(1,8):
   #print("Count, DataSet, Class, Model, Accuracy, Diff, ROC, Precision, Recall , F1, Cohen Kappa, Time Taken")
   for j in range(1,2):
        for k in range(1,2):
            NewX = 'DST'+str(i)
            NewY = 'CLT'+str(j)
            NewML = 'MLT'+str(k)

            NewX = eval(NewX)
            NewY = eval(NewY)
            NewML = eval(NewML)
            
            train_X= 'train_X'+str(i)
            train_Y='train_Y'+str(j)
            model_no='model'+str(k)
 
            train_X = eval(train_X)
            train_Y = eval(train_Y)
            model_no = eval(model_no)
            
            train_X=np.round_(train_X,decimals = 3)
            train_Y=np.round_(train_Y)
            holder = sys.stdout
            sys.stdout = open(outfile, "a")
            print(count,",",NewX,",",NewY,",",NewML,",", end=' ')
            Fold_10Looper(model_no,train_X,train_Y)
            count+=1
            sys.stdout = holder
            print(count)
            
            #logger.info(count)



sys.stdout.close()
print("Bub Bye World!")