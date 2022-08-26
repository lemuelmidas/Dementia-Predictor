import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("dementia_dataset.csv")
df.head()

#DATA CLEANING AND EXPLORATORY DATA ANALYSIS

#df.columns
#df.info()
#df.describe()
#df.corr()
#sns.heatmap(df.corr(), annot= True, cmap= 'Reds')
#df['Subject ID'].unique()
# Subject_ID is dropped because it is unifor across all instances.
df=df.drop('Subject ID', axis= 1)
#df.head()
# MRI_ID is dropped because it is unifor across all instances.
#df['MRI ID'].unique()
df=df.drop('MRI ID', axis= 1)
#df.head()
#df['Group'].unique()
#Label Encoding of 'Group' Feature

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df['Group']= le.fit_transform(df['Group'])
#df['Group'].unique()
#df.head()
#df['Visit'].unique()
#df['M/F'].unique()
#Binary Encoding of df['M/F']
df['M/F']= df['M/F'].apply(lambda x: 1 if x == 'M' else (0 if x == 'F' else None))
#df['M/F'].unique()
#df['Hand'].unique()
#the Hand column is dropped because its instances are the same for all rows
df= df.drop('Hand',axis=1)
#df.head()
#df['Age'].unique()
#df['EDUC'].unique()
#df['SES'].unique()
#df['SES'].mode()
#Replacing the nan values with the mode=2, since the mean is almost 2
df['SES']= df['SES'].replace(np.nan, 2)
#df['SES'].unique()
#df['SES'].value_counts()
#df['MMSE'].unique()
#print(df['MMSE'].mode())#,"\n",
#print(df['MMSE'].mean())#,"\n",
#print(df['MMSE'].value_counts())
#print(df['MMSE'].mode())#,"\n",
#print(df['MMSE'].mean())#,"\n",
#print(df['MMSE'].value_counts())
#df['MMSE'].isnull().sum()
#Replacing the nan values with the mode=30, since nan values are 2 and mode =30.
df['MMSE']= df['MMSE'].replace(np.nan, 30)
#df['MMSE'].unique()
#df['CDR'].unique()
#df['eTIV'].unique()
#df['nWBV'].unique()
#df['ASF'].unique()
#df.head(10)
X= df.drop('Group', axis=1)
y=df['Group']
#print(X.shape, y.shape)
#BASE MODEL
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.3, random_state= 48 )
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn import metrics
import random as rd

def model_evaluation(X,y):
    X_train,X_test, y_train, y_test= train_test_split(X, y, test_size= 0.3, random_state= 48 )
    models = [("Naive Bayes", GaussianNB()),
          ("KNN", KNeighborsClassifier()),
          ("DTC", DecisionTreeClassifier()),
          ("SGD", SGDClassifier()),
          ("Ada", AdaBoostClassifier()),
          ("RFC",RandomForestClassifier()),
          ("Extra:",ExtraTreesClassifier()),
          ("XGB:", xgb.XGBClassifier())
              
              
         ]
    means = []
    stds = []

    for name, classifier in models:
        scores = []
        for _ in range(10):
            model = classifier
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            Accuracy=metrics.accuracy_score(y_test,y_pred)

            scores.append(Accuracy)

        means.append(np.mean(scores))
        stds.append(np.std(scores))
        
    Total = list(zip(means,stds))
    hsh = {}
    for j in range(len(models)):
        hsh[models[j][0]] = list(Total[j])
        
        
    data = pd.DataFrame(hsh)
    db1=data.transpose()
    db1.columns=['mean','std']
    
    return db1
a=model_evaluation(df.drop('Group', axis=1),df['Group'])
#print(a)

model_xgb= xgb.XGBClassifier()
model_xgb.fit(X_train, y_train)
pred= model_xgb.predict(X_test)

import pickle
#filename='model_xgb.pkl'
#pickle.dump(model_xgb, open(filename, 'wb'))

#model_columns=list(X.columns)
#with open('columns.pkl', 'wb') as file:
#    pickle.dump(model_columns, file)
