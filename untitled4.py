import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =pd.read_csv('headbrain.csv')
x=data["Head Size(cm^3)"].values
y=data["Brain Weight(grams)"] .values

a=np.mean(x)
b=np.mean(y)

upper=0
lower=0

for i in range(0,len(x)):
    upper+=(x[i]-a)*(y[i]-b)
    lower+=(x[i]-a)**2
res=upper/lower

res2=b-res*a

x1=data.iloc[:,2:3].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x1,y)
m=regressor.coef_
c=regressor.intercept_
print(m)
print(c)

