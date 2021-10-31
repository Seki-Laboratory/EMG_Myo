import numpy as np
import matplotlib.pyplot as plt
import csv


LOAD_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/law2/"
SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/law3/"
JES = "9"

for i in range(0,51):
    CSVList=np.loadtxt(LOAD_DATA_DIR_PATH+JES+"/"+JES+'_emgdata'+str(i)+'.csv', delimiter=',')
    # img_gray = np.array(RMSList [:,0:8], dtype = np.uint8)
    img1 = np.array(CSVList [10:50,:], dtype = np.uint8)
    img2 = np.array(CSVList [51:91,:], dtype = np.uint8)
    np.savetxt(SAVE_DATA_DIR_PATH+JES+"/"+JES+'_emgdata1_'+str(i)+'.csv', img1, delimiter=',', fmt='%d')
    np.savetxt(SAVE_DATA_DIR_PATH+JES+"/"+JES+'_emgdata2_'+str(i)+'.csv', img2, delimiter=',', fmt='%d')

print("完了")
