from sklearn.datasets import load_digits
digits=load_digits()
images=digits['images']
import  matplotlib.pyplot as plt
image= images[25,:,:]
plt.imshow(image,cmap='gray')
targets=digits["target"]
targets[25]