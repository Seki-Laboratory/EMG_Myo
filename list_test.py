import numpy as np
from numpy.core.defchararray import array
from numpy.lib.function_base import gradient


arr = np.zeros(8,dtype = int)
result = np.zeros(8,dtype = int)
arr1 = np.array([1,1,1,1,1,1,1,1])


for i in range(3):
    result += arr1
    print(result)

print(result)

result = result/3

print(result)

# for i in range(20):
#     arr = np.append(arr, np.array([[1, 2, 3,5,6,8,9,3]]), axis=0)

# print(arr)
