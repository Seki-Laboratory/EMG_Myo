import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

RMSList=np.loadtxt('RMSdata.csv', delimiter=',')
element = RMSList [:,0:8]
label = np.ravel(RMSList[:,8:9])
rms_df = pd.DataFrame(element, columns=["e1","e2","e3","e4","e5","e6","e7","e8"])
rms_target_data = pd.DataFrame(label, columns=["label"])

X_train, X_test, Y_train, Y_test = train_test_split(rms_df,rms_target_data)

print(X_test)
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(rms_df, rms_target_data)

test = np.array([[15,1,2,5,3,5,7,1]])
# test = pd.DataFrame(test, columns=["e1","e2","e3","e4","e5","e6","e7","e8"])
print(knn.predict(test))
print(knn.predict(test))
print(knn.predict(test))
print(knn.predict(test))
print(knn.predict(test))
print(knn.predict(test))
print(knn.predict(test))





