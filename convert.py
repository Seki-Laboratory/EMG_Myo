import numpy as np
import matplotlib.pyplot as plt
import csv


LOAD_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/law/"
SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/law2/"
JES = "9"

for i in range(0,51):
    RMSList=np.loadtxt(LOAD_DATA_DIR_PATH+JES+"/"+JES+'_emgdata'+str(i)+'.csv', delimiter=',')
    img_gray = np.array(RMSList [:,0:8], dtype = np.uint8)
    # print(img_gray)
    # with open(SAVE_DATA_DIR_PATH+JES+"/"+JES+'_emgdata'+str(i)+'.csv', 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n') # 行末は改行
    #     writer.writerow(img_gray)
    np.savetxt(SAVE_DATA_DIR_PATH+JES+"/"+JES+'_emgdata'+str(i)+'.csv', img_gray, delimiter=',', fmt='%d')

    # with open(SAVE_DATA_DIR_PATH+JES+"/"+JES+'_emgdata'+str(i)+'.csv') as f:
    #     print(f.read())

