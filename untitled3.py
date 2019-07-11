
import numpy as np
import matplotlib.pyplot as plt

x=np.arange(1,11)
y=x**2
plt.plot(x,y,'r--',label="first")
plt.legend()
plt.title("abcd")
plt.xlabel("efgh")
plt.show()