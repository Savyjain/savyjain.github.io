import cv2
import numpy as np
import matplotlib.pyplot as plt


image =cv2.imread('nina.jpg')
imRed=image[:,:,2]
imBlue=image[:,:,0]
plt.subplot(2,2,1)

plt.imshow(imRed,cmap="gray")
plt.title("Red Component")

plt.subplot(2,2,2)

plt.imshow(imBlue,cmap="gray")
plt.title("Blue Component")

plt.subplot(2,2,3)

plt.imshow(image[:,:,1],cmap="gray")
plt.title("Green Component")

plt.subplot(2,2,4)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title("Original")




#logo=cv2.imread('logo.jpg',-1)
#plt.imshow(logo)
#newimage=logo+image
#newimage =  image+(logo*0.2)
#plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))





face = image[114:389,111:380,:]
#plt.imshow(face[::5,::5,:])

plt.imshow(face)
resizedImage=cv2.resize(face,(50,50))
plt.imshow(resizedImage)



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
