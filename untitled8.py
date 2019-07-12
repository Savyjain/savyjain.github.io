import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =pd.read_csv('insurance.csv')

data.columns
x=data.iloc[:,:6].values
y=data.iloc[:,6].values

from sklearn.preprocessing import LabelEncoder
lEncodergender= LabelEncoder()

x[:,1]=lEncodergender.fit_transform(x[:,1])

lEncodersmoker= LabelEncoder()

x[:,4]=lEncodersmoker.fit_transform(x[:,4])

lEncoderregion= LabelEncoder()

x[:,5]=lEncoderregion.fit_transform(x[:,5])

from sklearn.preprocessing import OneHotEncoder
ohEncoder = OneHotEncoder(categorical_features=[5]) #assigning no of dummy variables

x=ohEncoder.fit_transform(x).toarray() #dummy variables are created
x=x[:,1:] #1st dummy column is removed

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)

score=regressor.score(x_test,y_test)


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test,y_pred)**(1/2)