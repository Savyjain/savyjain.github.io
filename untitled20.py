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
image=cv2.putText(image,"Nina",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)