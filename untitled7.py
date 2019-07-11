import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('headbrain.csv')
x=data.iloc[:,2].values
y=data.iloc[:,3].values
z=np.mean(x)
a=np.mean(y)
sum1=0
div=0
for i in range(0,len(y)):
    sum1+=(x[i]-z)*(y[i]-a)
    div+=(x[i]-z)**2
b1=sum1/div #slope
b0=a-b1*z   #intersept
print(b1)
print(b0)
x1=data.iloc[:,2:3].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x1,y)
m=regressor.coef_
c=regressor.intercept_
print(m)
print(c)
y_pred=regressor.predict(x1)
sst=0
ssr=0
for i in range(0,len(x)):
    sst+=(y[i] - a)**2
    ssr+=(y[i] - y_pred[i])**2
R2=1-(ssr/sst)
print("Score=",R2)
print(regressor.score(x1,y))


