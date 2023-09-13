# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 04:16:13 2023

@author: arpita
"""

#importing the libararies
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

#reading the data from your files
data=pd.read_csv('advertising.csv')
data.head()

#to visualize data
fig , axs = plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])

#creating x and y for linear regressoin
feature_cols=['TV']
x=data[feature_cols]
y=data.Sales

#importing linear regression algo
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
#fit is used for fitting the entire data in the algo 


print(lr.intercept_)
print(lr.coef_)
"""considering if we are trying to predict the amount of sales got from the 
selling of tv,radio,newspaper(profit)
y=a+bx
y=Sales(independent variable)
x=TV,Radio,Newspaper(dependent variable)
a=intercept
b=coef(of x)"""


#creating a dataframe with min and max of the table
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
#is used for finding the minimum and maximum value present in the column
X_new.head()

preds=lr.predict(X_new)
preds
"""lr.predicts is used for predicting (in this case) what will be the sales
 if a given amount of money is invested on the product"""

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(X_new,preds,c='red',linewidth=3)#bestfit line
"""Least Squares Regression line -if your data 
shows linear regression between x and y variable
,you will want to find the line that best fits
this linear relationship.That line is calle 
REGRESSION LINE and has the eq.y=a+bx .
The least square line is the line that makes 
the vertical distance from the data points to the 
regression line as small as possible """

#finding the summary and configuration 
import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales ~ TV',data=data).fit()
lm.conf_int()

lm.pvalues#finding the probability values
"""p-value is the probability obtained result as extreme as the 
observed result of statistics hypothysis,assuming that the null 
hypothesis is correct .The p-value is used alternative to rejection
points the smallest level of significance at which null hypothesis
would be rejected.A smaller p-value means that there is a stronger
evidence in favor of the alternative hypothesis"""
 

#finiding the r-square value 
lm.rsquared

feature_cols=['TV','Radio','Newspaper']
x=data[feature_cols]
y=data.Sales

lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)


lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()           



