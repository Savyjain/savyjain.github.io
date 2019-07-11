import numpy as np
mat = np.ones((8,8))
for i in range(0,8):
  for j in range(0,8):
    if(i+j)%2==0:
      mat[i,j]=0
print(mat)

