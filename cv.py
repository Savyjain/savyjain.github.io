import cv2
import numpy as np
import matplotlib.pyplot as plt

#black=np.zeros((48,48))
#black[::4]=255
#plt.imshow(black,cmap="gray")
#a="abcdefghijklmnopqrstuvwxyz"
#cv2.imshow('frame Name',black)
#cv2.waitKey(0)



#image =cv2.imread('lena.jpg',0)
#cv2.imshow('title',image)


image =cv2.imread('nina.jpg',-1)
imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.subplot(1,2,1)
plt.imshow(imageGray,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))