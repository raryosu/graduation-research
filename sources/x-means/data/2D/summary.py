import sys
import numpy as np
import matplotlib as plt

data = np.loadtxt(sys.argv[1], delimiter=",")
print("mean: ", np.mean(data, axis=0))
print("var: ", np.var(data, axis=0))

k = data[:,0]

for i in range(int(np.min(k)), int(np.max(k)+1.0)):
    num = len(np.where(k==i))
    print(i, num)

