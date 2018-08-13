#This is file has been copy pasted from kaggle jupyter notebook. May contain minute errros while copy pasting. Link to the notebook can be found in readme file.

import random
import time
import sys
import numpy as np
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import contour
import seaborn as sns
from IPython.display import display
from scipy import sparse
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from numpy import inf
from numpy.random import uniform
from numpy.random import randint

import os
import os.path

pd.set_option('max_columns',None)
pd.options.display.width = 2000
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

print(os.listdir("../input"))
mpl.rcParams['agg.path.chunksize'] = 10000
%matplotlib inline

def printruntime(label,startTime):
    print(label, (time.time()-startTime))

runtime = time.time()
df_cc = pd.read_csv("../input/creditcard.csv", low_memory=False)
printruntime("Load Time",runtime)

display(df_cc[0:20])
print("Any null values?",df_cc.isnull().any().any())
fraud_count = len(df_cc[df_cc["Class"]==1])
total_count = len(df_cc)
print("Number of fraud cases",fraud_count,"out of total",total_count,"which is",round((fraud_count/total_count)*100,2),"%")

df_fraud = df_cc[df_cc["Class"]==1]
df_auth = df_cc[df_cc["Class"]==0]

#selecting columns that show maximum mean difference between fraud and auth transactions
cols = []
for i in range(1,29):
    cols.append("V"+str(i))
cols.append("Amount")
df_ranges = pd.DataFrame({"Auth_Mins":df_auth.min(),"Auth_Maxs":df_auth.max(),"Auth_Mean":df_cc.mean(),"Fraud_Mins":df_fraud.min(),"Fraud_Maxs":df_fraud.max(),"Fraud_Mean":df_fraud.mean()},index=cols)
df_ranges["mean_diff_abs"] = np.abs(df_ranges["Auth_Mean"] - df_ranges["Fraud_Mean"])
df_ranges.sort_values(by="mean_diff_abs",ascending=False, inplace=True)
display(df_ranges)

# selecting columns whose mean differe by atleast margin 4 between auth and fraud transaction
columns = list(df_ranges[1:15].index) #starting from 1 instead of 0 to exclude Amount.
columns.append("Class") # we need this to test our prediction
print("Columns that could be used for Anomaly detection : ",columns) 

#checking selected columns show any different behaviours for fraud and auth transactions
#plotting only subset of dataset as full plot would take a long time
df_temp = pd.concat([df_fraud.sample(n=200),df_auth.sample(n=300)],ignore_index=True,verify_integrity=True)
classes = ["Authentic","Fraud"]
sns.set(style="ticks", color_codes=True)
pair_grid = sns.pairplot(data=df_temp[columns[0:5]+["Class"]], hue="Class",kind='scatter',palette={0:'g',1:'r'},markers=['o','x'])
pair_grid.add_legend()

del df_temp

#Splitting whole data into training,cross validation and test sets. Training set has only authentic transactions,
#Cross validation & Test set has both authenctic and fraud transactions
df_train,df_split2 = train_test_split(df_auth[columns],train_size=0.6)
tr_temp, te_temp = train_test_split(df_fraud[columns],train_size=0.5)
df_cv1,df_cv2 = train_test_split(df_split2,train_size=0.5)
df_cv = pd.concat([df_cv1,tr_temp],ignore_index=True,verify_integrity=True)
df_test = pd.concat([df_cv2,te_temp],ignore_index=True,verify_integrity=True)

del df_split2,tr_temp, te_temp,df_cv1,df_cv2

#randomize df rows
df_cv = df_cv.sample(frac=1).reset_index(drop=True)
df_test = df_test.sample(frac=1).reset_index(drop=True)

#calculating mu & sig for training set
m_tr = len(df_train)
x_mat_tr = np.matrix(df_train.values)
mu = np.squeeze(np.asarray(x_mat_tr[:,0:-1].sum(0) / m_tr)) #1 X n
sig_sq = np.squeeze(np.asarray(np.square(x_mat_tr[:,0:-1] - mu).sum(0) / m_tr)) # 1 X n
sig = np.sqrt(sig_sq)
#print("Shape of mu & sig_sq train",mu.shape,sig_sq.shape)

#fig = plt.plot(df_cc["V3"],np.zeros(len(df_cc)), marker='x',kind='scatter')
#fig.set_figheight(50)
#fig.set_figwidth(50)
#plt.show()

def checkIfColumnsGaussian(df_cc):
    m = len(df_cc)
    n = len(df_cc.columns)-1
    rows = math.ceil(n / 3)
    #print("columns=",n,"rows=",rows)
    x_cc = np.matrix(df_cc.values)
    mu_cc = np.squeeze(np.asarray(x_cc[:,0:-1].sum(0) / m)) #1 X n
    sig_sq_cc = np.squeeze(np.asarray(np.square(x_cc[:,0:-1] - mu_cc).sum(0) / m)) # 1 X n
    sig_cc = np.sqrt(sig_sq_cc)
    
    fig,ax = plt.subplots(rows,3)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    count = 0;
    for row in ax:
        for subplot in row:
            if(count == 14):
                break
            xvalues = np.squeeze(np.asarray(np.sort(x_cc[:,count])))
            #print(col0.shape,mu[ind],sig_sq[ind],col0[0],col0[-1])
            x_axis = np.unique(xvalues)#np.linspace(col0[0],col0[-1],100000)
            y_axis = norm(mu_cc[count],sig_cc[count]).pdf(x_axis)
            subplot.plot(x_axis, y_axis)
            #subplot.scatter(x_axis,np.zeros(len(x_axis)),marker='x')
            subplot.set_xlabel(columns[count])
            subplot.set_ylabel('PDF') 
            count = count + 1
    plt.show()
    
        
checkIfColumnsGaussian(df_cc[columns])
#contour(x_mat_tr_p)
# All features seem to have bell curve using calculated mu & sig. Proceed with prediction

#This is the threshold on basis of which we decide if a give transaction is authentic or fraud. This is guessed looking at
#cross validation PDF values of fraud transactions
epsilon = 2.5 * (10**-10) #1.2 * (10**-10) #1.3 * (10**-10) 15 cols #0.5 * (10**-13) with 20 cols,14

def predict(df):
    global sig
    global mu
    global epsilon
    #global df_fraud
    
    x = np.matrix(df[columns].values)
    px = norm(mu,sig).pdf(x[:,0:-1])
    p = np.prod(px,axis=1)
    pdc = np.column_stack((p,x[:,-1]))#np.matrix(np.zeros((p.shape[0],2)))
    df_pdc = pd.DataFrame(pdc,columns=["PD","Class"])
    #print("Max/Min for Probability density values for fraud cases",df_pdc[df_pdc["Class"]==1]["PD"].max(),df_pdc[df_pdc["Class"]==1]["PD"].min(), np.sort(df_pdc[df_pdc["Class"]==1]["PD"])[-2])
    pred = np.zeros(len(df_pdc))
    pred[df_pdc["PD"] < epsilon] = 1
    prec,recall,fbeta,support = precision_recall_fscore_support(x[:,-1],pred,average='binary')
    #P = Tp/(Tp+Fp), R = Tp/(Tp+Fn), F1 = 2*(PR)/(P+R)
    print("Precision",prec,"recall",recall,"accuracy",accuracy_score(x[:,-1],pred),"fbeta",fbeta)
    total1s = (x[:,-1]==1).sum()
    total0s = (x[:,-1]==0).sum()
    #print("Total",len(x),"Actual 1's",total1s,"Pred 1's",(pred==1).sum(),"Actual 0's",total0s,"Pred 0",(pred==0).sum())
    confMat = confusion_matrix(x[:,-1],pred)
    print(confMat)
    print("Summary: Recall % = ",round(recall*100,2),", False positive out of all negatives%",round(confMat[0,1]/total0s*100,2), ", False Negative out of all positive%",round(confMat[1,0]/total1s*100,2))
    
print("\n******Cross Validation Results******")
predict(df_cv)
print("\n******Test Set Results******")
predict(df_test)
#predict(df_cc)
    
printruntime("Total Runtime",runtime)
