import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =pd.read_csv('headbrain.csv')
x=data["Head Size(cm^3)"].values
y=data["Brain Weight(grams)"] .values

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
b=np.mean(y)
c=regressor.predict(y[i])
ssr,sss=0,0
for i in range(0,len(x)):
    ssr+=(y[i]-b)**2
    sss+=(y[i]-c)**2
r=(1- ssr/sss)