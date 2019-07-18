import cv2
import numpy as np
import matplotlib.pyplot as plt

faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
facedata=[]
facecount=0
while True:
    ret,image=cap.read()
    
    faceGray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(faceGray,1.3,5)

    for (x,y,w,h) in faces:
        cropedface=image[y:y+h ,x:x+w ,:]
        resizedface=cv2.resize(cropedface,(50,50))
        if facecount % 10 ==0 and len(facedata)<=20:
           facedata.append(resizedface)
    facecount+=1 
       
    cv2.imshow('original', image)
    if cv2.waitKey(1)==27 or len(facedata)>20:
        
        break   
cap.release()
cv2.destroyAllWindows()

facedata =np.array(facedata)
np.save('Savy',facedata)