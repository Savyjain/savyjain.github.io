import numpy as np
import pandas as pd

data = pd.read_csv('Social_Network_Ads.csv')
data.head()
x = data.iloc[:,1:4].values
y = data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
lEncoder = LabelEncoder()
x[:,0] = lEncoder.fit_transform(x[:,0])

from sklearn.preprocessing import StandardScaler
sScaler = StandardScaler()
x = sScaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=0)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3,)
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
print(clf.score)


y_pred = clf.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(clf.score(x_test,y_test))


v = ['Female',32,150000]
v = np.array(v)
v = v.reshape((1,-1))
v[:,0] = lEncoder.transform(v[:,0])
scaledV = sScaler.transform(v)
clf.predict(scaledV)

#tofind distances between two points
def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(testinput,x,y,k):
    numRows = x.shape[0]#data=x,purchased=y
    dist = []
    for item in range(numRows):
        dist.append(distance(testinput,x[item]))
    dist = np.array(dist)
    index = np.argsort(dist) #to sort and to find index of the elemnts
    sortedLabels = y[index][:k]
    count = np.unique(sortedLabels,return_counts=True)
    return count[0][np.argmax(count[-1])]
    
knn(scaledV,x_train,y_train,3)
yPredNew = []
for i in range(0,len(x_test)):
    yPredNew.append(knn(x_test[i],x_train,y_train,3))   #for multiple inputs
confusion_matrix(y_test,yPredNew)
        

