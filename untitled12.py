import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
iris=load_iris()

x= iris["data"]#features
y=iris.target#labels
targetnames=iris.target_names


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()

clf.fit(x_train,y_train)

from sklearn.externals import joblib
joblib.dump(clf, 'irisPred.sav')

from sklearn.externals import joblib
model=joblib.load('irisPred.sav')

model.predict([[5.8,2.8,5.1,2.4]])
y_pred=clf.predict(x_test)

score=clf.score(x_test,y_test)

#iris.columns
userinput =eval(input('enter the data :'))
userinput=list(userinput)

pred=clf.predict([userinput])

print("the flower is",targetnames[pred])