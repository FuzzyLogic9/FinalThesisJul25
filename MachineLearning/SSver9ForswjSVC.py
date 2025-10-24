from dask.dataframe.methods import values
from numpy.lib.function_base import average
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning,ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from numpy import loadtxt
#from sklearnex import patch_sklearn 
#patch_sklearn()

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection._validation import cross_val_score 
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, f1_score,roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

import pandas as pd
import numpy as np
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

import sklearn
#print(sklearn.__version__)
 
#model1 = RandomForestClassifier() ##
#model2 = KNeighborsClassifier(n_neighbors=5) ##
#model3 = GaussianNB() ##
#model4 = LogisticRegression(solver="liblinear") ##
#model5= MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1,max_iter=1000) ##why does it return 3?
model1 = SVC(gamma='auto',probability=True)
#model1 = svm.SVC(kernel="linear", C=0.025,probability=True,decision_function_shape='ovr')
#model2 = svm.SVC(kernel="linear", C=0.025,probability=True,decision_function_shape='ovo')
#model3 = svm.SVC(kernel="linear", C=1,probability=True,decision_function_shape='ovr')
#model4 = svm.SVC(kernel="linear", C=1,probability=True,decision_function_shape='ovo')
#model5 = svm.SVC()

#>>> clf = CalibratedClassifierCV(svm)
#>>> clf.fit(X_train, y_train)

#model7 = svm.SVC(decision_function_shape='ovo')
MLT1="SVCGAMMA"



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



def RunClass(model_no,train_x,test_x,train_y,test_y):
   
   
  
    model = model_no
    # fit the model on the whole dataset
    Model_Fit = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
    
    #Model_Fit = CalibratedClassifierCV(model)
    Model_Fit=Model_Fit.fit(train_x,train_y)
    #clf.fit(X, y)
    # make a single prediction
    i = 0
    a = 0
    #CalibratedClassifierCV(base_estimator=model(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
   #decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
   #max_iter=-1, probability=False, random_state=None, shrinking=True,
   #tol=0.001, verbose=False),cv=3, method='sigmoid')
#>>> clf.predict_proba(X_test)
#array([[0.02352877, 0.64021213, 0.33625911],
    The_Prediction =  Model_Fit.predict(test_x)
    Predictions_proba = Model_Fit.predict_proba(test_x)

    test_y_arr= np.array(test_y)
    #print(test_y_arr.shape)
    pred_probs_arr= np.array(Predictions_proba)

   
    #accuracy
    pred_acc = Model_Fit.score(test_x, test_y)

    #ROC
    
    #roc_auc_scores = roc_auc_score((test_y), Predictions_proba[:,1])
    roc_auc_scores = roc_auc_score((test_y), Predictions_proba[:,1], multi_class='ovo')
    #roc_auc_scores = roc_auc_score((test_y), Predictions_proba[:,1], multi_class='ovr')
   
    
    preci = precision_score(test_y, The_Prediction)
    recall = recall_score (test_y, The_Prediction)
    fone = f1_score(test_y, The_Prediction)


    cohen_kappascore = cohen_kappa_score(test_y, The_Prediction)




    #diff = ((The_Prediction)-(test_y)).tolist()
    diff = ((pred_acc)-(test_y)).tolist()
    return(pred_acc,((diff.count(0)) / len(diff)),(roc_auc_scores),preci, recall, fone, cohen_kappascore)
    




def Fold_10Looper(model_no,train_X,train_Y):
    
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
      
  
    print(np.mean(totalacc),",",np.mean(totaldiff),",",np.mean(totalroc),",",np.mean(totalpreci),",",np.mean(totalrec),",",np.mean(totalf1s),",",np.mean(totalcks))





items = []

infile = 'H:\\Scraping_Training_VSC\\Training_Exp1\\DatafromWOBtots\\WOB39\\NEWDBSA\\FinalCodes\\PipelineEntryFileFeb22MCNH_OR.csv'
#infile="C:\\Users\\1943069\\Scraping_Training_VSC\\Training_Exp1\\PipelineEntryFileFeb22MCNH1.csv"

#infile = 'H:\\Scraping_Training_VSC\\Training_Exp1\\DatafromWOBtots\\WOB39\\NEWDBSA\\FinalCodes\\PipelineEntryFileFeb22MCNH1.csv'

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
# [30] Humannotated Toxic Class Average || [31]	Humannotated Toxic Class Average Wholed [32] Humannotated Toxic Class || [3] Multi-Class

outfile = "C:\\Users\\1943069\\Scraping_Training_VSC\\Training_Exp1\\OutputforloopallwSVC_OR.csv"



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
train_Y1 = df.iloc[:,26].values
    #train_Y= train_Y_with_only_Aggression
CLT1="AggClass"
#print("2. Train_Y_with_only_Aggression_Average ") 
#train_Y2 = df.iloc[:,25].values
    #train_Y = train_Y_with_only_Aggression_Avg

#print("3. Train_Y_with_only_Attacks")  
train_Y2 = df.iloc[:,29].values
    #train_Y = train_Y_with_only_Attacks
CLT2="AttClass"
#print("4. Train_Y_with_only_Attacks_Average")
#train_Y4 = df.iloc[:,28].values
    #train_Y = train_Y_with_only_Attacks_Avg

#print("5. Train_Y_with_only_Toxicity") 
train_Y3 = df.iloc[:,32].values
    #train_Y = train_Y_with_only_Toxicity
CLT3="ToxClass"
#print("6. Train_Y_with_only_Toxicity_Average")  
#train_Y6 = df.iloc[:,31].values
  
#print("7. Train_Y_with_Multiclass Comb")  
train_Y4 = df.iloc[:,33].values
CLT4 ="MutiClass"



#Do all your dataset, class model choice and measurements with ALL option

count =1
print("Count, DataSet, Class, Model, Accuracy, Diff, ROC, Precision, Recall , F1, Cohen Kappa")
for i in range(1,8):
    for j in range(1,4):
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



sys.stdout.close()
print("Bub Bye World!")