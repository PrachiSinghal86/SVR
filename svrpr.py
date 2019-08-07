# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 00:20:43 2019

@author: user
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('Position_Salaries.Csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
X_pr=X
X_pr[9,0]=6.5
#data splitting

"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
"""
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#Fitting  regression model to dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)
#Predicting polynomial regression
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#visualizing polynomial regression

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Salary Prediction')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()