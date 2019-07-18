import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =pd.read_csv('Social_Network_Ads.csv')

data.columns
x=data.iloc[:,1:4].values
y=data.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder
lEncoder= LabelEncoder()

x[:,0]=lEncoder.fit_transform(x[:,0])#x is considered from 1st column and 0th column is not included hence here we take 1st column itself as 0th column

from sklearn.preprocessing import StandardScaler
sscalar=StandardScaler()
x=sscalar.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

classifier.fit(x_train,y_train)

y_pred= classifier.predict(x_test)

c=0

for i in range (len(y_pred)):# to identify how many prediction are correct
    if y_pred[i]==y_test[i]:
            c+=1
print('correct prediction',c)

from sklearn.metrics import confusion_matrix# to identify how many prediction are correct
cm= confusion_matrix(y_test,y_pred)
score=classifier.score(x_test,y_test)