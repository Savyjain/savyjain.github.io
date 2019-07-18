import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_digits
digits=load_digits()

#images=digits.images
#image=images[100]
#plt.imshow(image,cmap='gray')


#position=1
#for k in range(1,11):
 #   plt.subplot(5,2,k)
  #  plt.imshow(images[k],cmap='gray')
   # plt.title(digits.target[k])
   
x=digits.data
y=digits.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

score=clf.score(x_test,y_test)


from sklearn.metrics import confusion_matrix# to identify how many prediction are correct
cm= confusion_matrix(y_test,y_pred)

c=0

for i in range (len(y_pred)):# to identify how many prediction are correct
    if y_pred[i]==y_test[i]:
            c+=1
print('correct prediction',c)