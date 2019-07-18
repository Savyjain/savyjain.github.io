import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =pd.read_csv('Position_Salaries.csv')

data.columns
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values


plt.scatter(x,y)


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x,y)

plt.plot(x,regressor.predict(x),'red')

from sklearn.preprocessing import PolynomialFeatures

score=0

d=2
ypred=[]
while score<=0.9997:#to find at what degree score=0.999
    polyfeatures=PolynomialFeatures(degree=d)
    xpoly=polyfeatures.fit_transform(x)
    regressor1= LinearRegression()
    regressor1.fit(xpoly,y)
    ypred.append(regressor1.predict(xpoly))
    score=regressor1.score(xpoly,y)
    d=d+1

plt.plot(x,regressor1.predict(xpoly),'green')

position=1
for k in ypred:
    plt.subplot(2,2,position)
    plt.scatter(x,y,color='blue')
    plt.plot(x,k,'green')
    position+=1