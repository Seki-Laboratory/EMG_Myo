import numpy as np
import matplotlib.pyplot as plt
import csv

JES = "ko"
LOAD_DATA_DIR_PATH = "C:/Users/usui0/Desktop/emg_data"
SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/data/"+JES


for i in range(1000,10100,1):
    CSVList=np.loadtxt('C:/Users/usui0/Desktop/emg_data/'+JES+'/'+JES+'_emgdata0.csv', delimiter=',')
    # img_gray = np.array(RMSList [:,0:8], dtype = np.uint8)
    img1 = np.array(CSVList [i-1000:i,:],dtype = np.float)
    img  = img1*1000
    np.savetxt(SAVE_DATA_DIR_PATH+"/"+str(i-1000)+'.csv', img, delimiter=',', fmt='%d')
    print(i)


print("完了")
