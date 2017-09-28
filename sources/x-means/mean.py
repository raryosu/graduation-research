import sys
import numpy as np

data = np.loadtxt(sys.argv[1], delimiter=",")
print("mean: ", np.mean(data, axis=0))
print("var: ", np.var(data, axis=0))
