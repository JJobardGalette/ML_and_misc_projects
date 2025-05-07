import math
from PIL import Image
import numpy as np
import os
W = 201
L = 201
n_iter = 50
M = np.empty((W,L),int)
#print(M)

def norm(a,b):
    a = 2*(a-(W+1)/2+1)/W-0.3
    b = 2*(b-(L+1)/2+1)/L
    return complex(a,b)

for x in range(W):
    for y in range(L):
        z = norm(x,y)
        #print(z)
        M[y,x] = 8
        for j in range(n_iter):
            z = z**2 + norm(x,y)
            if abs(z) > 2:
                M[y,x] = 1
                break

np.savetxt('export2.txt',M,fmt = '%d',delimiter = '')
print(M)
