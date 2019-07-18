
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)

data.columns
userinput =input('enter age,sex,bmi,children,smoker,region:')
userinput=userinput.split(',')
#lEncodergender.transform(['Male'])
#lEncodergender.transform(['Female'])

u=[]
for item in userinput:
    try:
        u.append(float(item))
    except:
        u.append(item)
    
u=np.array(u).reshape((1,-1))


u[:,1]=lEncodergender.transform(u[:,1])

u[:,4]=lEncodersmoker.transform(u[:,4])

u[:,5]=lEncoderregion.transform(u[:,5])

u=ohEncoder.transform(u).toarray() #dummy variables are created
u=u[:,1:] #1st dummy column is removed     
regressor.predict(u)
runfile('C:/Users/DELL/.spyder-py3/untitled8.py', wdir='C:/Users/DELL/.spyder-py3')
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

data.columns
userinput =input('enter age,sex,bmi,children,smoker,region:')
userinput=userinput.split(',')
#lEncodergender.transform(['Male'])
#lEncodergender.transform(['Female'])

u=[]
for item in userinput:
    try:
        u.append(float(item))
    except:
        u.append(item)
    
u=np.array(u).reshape((1,-1))


u[:,1]=lEncodergender.transform(u[:,1])

u[:,4]=lEncodersmoker.transform(u[:,4])

u[:,5]=lEncoderregion.transform(u[:,5])

u=ohEncoder.transform(u).toarray() #dummy variables are created
u=u[:,1:] #1st dummy column is removed     
regressor.predict(u)

## ---(Sat Jul 13 11:47:38 2019)---
runfile('C:/Users/DELL/.spyder-py3/untitled12.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled12.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled11.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled13.py', wdir='C:/Users/DELL/.spyder-py3')

## ---(Mon Jul 15 10:45:57 2019)---
runfile('C:/Users/DELL/app.py', wdir='C:/Users/DELL')
runfile('C:/Users/DELL/.spyder-py3/app1.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled12.py', wdir='C:/Users/DELL/.spyder-py3')
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
runfile('C:/Users/DELL/.spyder-py3/untitled12.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled121.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/app3.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled121.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled12.py', wdir='C:/Users/DELL/.spyder-py3')

from sklearn.externals import joblib
model=joblib.load('irisPred.sav')

model.predict([[5.8,2.8,5.1,2.4]])
y_pred=clf.predict(x_test)
from sklearn.externals import joblib
model=joblib.load('irisPred.sav')

model.predict([[5.8,2.8,5.1,2.4]])
y_pred=clf.predict(x_test)
runfile('C:/Users/DELL/.spyder-py3/untitled121.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/app3.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled12.py', wdir='C:/Users/DELL/.spyder-py3')
import cv2
runfile('C:/Users/DELL/.spyder-py3/untitled20.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/cv.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/cv_red.py', wdir='C:/Users/DELL/.spyder-py3')
cap = cv2.VideoCapture(0)
ret,img=cap.read()
cap.release()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
r,c,l=img.shape
output=np.zeros((r,c))

rMax=255
rMin=250
bMax=160
bMin=0
gMax=200
gMin=100

for i in range(0,r):
    for j in range(0,c):
        if img[i,j,0]<bMax and img[i,j,0]>bMin and img[i,j,1]<rMax and img[i,j,1]>rMin:
            output[i,j]=255
        else:
            output[i,j]=0
            
plt.imshow(output,cmap="gray")
cap = cv2.VideoCapture(0)
ret,img=cap.read()
cap.release()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
runfile('C:/Users/DELL/.spyder-py3/cv_red.py', wdir='C:/Users/DELL/.spyder-py3')

cap = cv2.VideoCapture(0)
ret,img=cap.read()
cap.release()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
r,c,l=img.shape
output=np.zeros((r,c))

rMax=255
rMin=250
bMax=160
bMin=0
gMax=200
gMin=100

for i in range(0,r):
    for j in range(0,c):
        if img[i,j,0]<bMax and img[i,j,0]>bMin and img[i,j,1]<rMax and img[i,j,1]>rMin:
            output[i,j]=255
        else:
            output[i,j]=0
            
plt.imshow(output,cmap="gray")
runfile('C:/Users/DELL/.spyder-py3/untitled20.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled22.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled20.py', wdir='C:/Users/DELL/.spyder-py3')
face=faceCascade.detectMultiScale(imGray,1.3,5)
x,y,h,w =face[0]
newFace= image[y:y+h , x:x+w ,:]

#plt.imshow(cv2.cvtColor(newFace,cv2.COLOR_BGR2RGB))

nfGray= cv2.cvtColor(newFace,cv2.COLOR_BGR2GRAY)
newFaceEdge=cv2.Canny(nfGray,30,200)
plt.imshow(newFaceEdge,cmap='gray')

plt.imshow(image)
image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
runfile('C:/Users/DELL/.spyder-py3/untitled20.py', wdir='C:/Users/DELL/.spyder-py3')
face=faceCascade.detectMultiScale(imGray,1.3,5)
x,y,h,w =face[0]
newFace= image[y:y+h , x:x+w ,:]
nfGray= cv2.cvtColor(newFace,cv2.COLOR_BGR2GRAY)
newFaceEdge=cv2.Canny(nfGray,30,200)
plt.imshow(newFaceEdge,cmap='gray')
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('nina.jpg')
#%matplotlib auto
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

imGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
#plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.imshow(imGray,cmap='gray')

face=faceCascade.detectMultiScale(imGray,1.3,5)
x,y,h,w =face[0]
newFace= image[y:y+h , x:x+w ,:]

#plt.imshow(cv2.cvtColor(newFace,cv2.COLOR_BGR2RGB))

nfGray= cv2.cvtColor(newFace,cv2.COLOR_BGR2GRAY)
newFaceEdge=cv2.Canny(nfGray,30,200)
plt.imshow(newFaceEdge,cmap='gray')

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
face=faceCascade.detectMultiScale(imGray,1.3,5)
x,y,h,w =face[0]
newFace= image[y:y+h , x:x+w ,:]
import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('nina.jpg')
#%matplotlib auto
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

imGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
#plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.imshow(imGray,cmap='gray')

face=faceCascade.detectMultiScale(imGray,1.3,5)
x,y,h,w =face[0]
newFace= image[y:y+h , x:x+w ,:]

face=faceCascade.detectMultiScale(imGray,1.3,5)
x,y,h,w =face[0]
newFace= image[y:y+h , x:x+w ,:]
nfGray= cv2.cvtColor(newFace,cv2.COLOR_BGR2GRAY)
newFaceEdge=cv2.Canny(nfGray,30,200)
plt.imshow(newFaceEdge,cmap='gray')

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
runfile('C:/Users/DELL/.spyder-py3/untitled20.py', wdir='C:/Users/DELL/.spyder-py3')
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
image=cv2.putText(image,"Nina",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
runfile('C:/Users/DELL/.spyder-py3/untitled22.py', wdir='C:/Users/DELL/.spyder-py3')

image=cv2.imread('image.jpg')
imGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imGray,1.15,5)

for face in faces:
    x,y,h,w=face
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))





image=cv2.imread('image.jpg')
imGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imGray,1.15,5)

for face in faces:
    x,y,h,w=face
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

image=cv2.imread('image.jpg')
imGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imGray,1.15,5)

for face in faces:
    x,y,h,w=face
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    image=cv2.putText(image,"Nina",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))


image=cv2.imread('image.jpg')
imGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imGray,1.15,5)

for face in faces:
    x,y,h,w=face
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    image=cv2.putText(image,"Nina",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
runfile('C:/Users/DELL/.spyder-py3/untitled22.py', wdir='C:/Users/DELL/.spyder-py3')
image=cv2.imread('image.jpg')
imGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imGray,1.15,5)

for face in faces:
    x,y,h,w=face
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
   
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
cap=cv2.VideoCapture(0)

while True:
    ret,image=cap.read()
    cv2.imshow('video', image)
    
    faceGray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(faceGray,1.3,5)
    for face in faces:
        x,y,h,w=face
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('video', image)
    if cv2.waitKey(1)==27:
        cap.release()
        cv2.destroyAllWindows()
        break
        
cap.release()
runfile('C:/Users/DELL/.spyder-py3/untitled22.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled20.py', wdir='C:/Users/DELL/.spyder-py3')
cap.release()
runfile('C:/Users/DELL/.spyder-py3/untitled22.py', wdir='C:/Users/DELL/.spyder-py3')
cap.release()
runfile('C:/Users/DELL/.spyder-py3/untitled23.py', wdir='C:/Users/DELL/.spyder-py3')
cap.release()
runfile('C:/Users/DELL/.spyder-py3/untitled23.py', wdir='C:/Users/DELL/.spyder-py3')
cap.release()
runfile('C:/Users/DELL/.spyder-py3/untitled23.py', wdir='C:/Users/DELL/.spyder-py3')
runfile('C:/Users/DELL/.spyder-py3/untitled24.py', wdir='C:/Users/DELL/.spyder-py3')
cap.release()
runfile('C:/Users/DELL/.spyder-py3/untitled24.py', wdir='C:/Users/DELL/.spyder-py3')
cap.release()
runfile('C:/Users/DELL/.spyder-py3/untitled24.py', wdir='C:/Users/DELL/.spyder-py3')
cap.release()