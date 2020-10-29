import numpy as np


arr = np.empty((0,8), int)
print(arr)
for i in range(20):
    arr = np.append(arr, np.array([[1, 2, 3,5,6,8,9,3]]), axis=0)

print(arr)
